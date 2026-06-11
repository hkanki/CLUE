# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random
import numpy as np

import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import utils
from utils import ActualSequentialSampler
from adapt.solvers.solver import get_solver
from uot_utils import BalancedBatchSampler
import time

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)
class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        # Subset なら元データの絶対インデックスを保持
        if hasattr(base_dataset, "indices"):
            self.indices = base_dataset.indices
        else:
            self.indices = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        base_idx = self.indices[i] if self.indices is not None else i
        return x, y, base_idx

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		self.dset = dset
		if dset.name == 'DomainNet':
			self.num_classes = self.dset.get_num_classes()
		else:
			self.num_classes = len(set(dset.targets.numpy()))
		self.train_idx = np.array(train_idx)
		self.model = model
		self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)

	def query(self, n):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb

	def train(self, target_train_dset, da_round=1, src_loader=None, src_model=None):
		"""
		Driver train method
		"""
		best_val_acc, best_model = 0.0, None
		
		train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
		if self.args.da_strat != 'a':
			labeled_indices = self.train_idx[self.idxs_lb]
			unlabeled_indices = self.train_idx[~self.idxs_lb]

			labeled_dataset = torch.utils.data.Subset(target_train_dset, labeled_indices)
			
			labels = torch.tensor([target_train_dset[i][1] for i in labeled_indices])


			indexed_dataset = labeled_dataset

			# BalancedBatchSampler の作成
			num_labeled = len(labeled_indices)
			tgt_sup_loader = torch.utils.data.DataLoader(
				indexed_dataset,
				batch_size=min(self.args.batch_size, num_labeled),
				shuffle=True,
				num_workers=0,
				drop_last=False
			)

			# tgt_sup_loader の作成（Sampler を明示指定）
			# tgt_sup_loader = torch.utils.data.DataLoader(
			#     indexed_dataset,
			#     batch_sampler=batch_sampler,
			#     num_workers=0
			# )
			# train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
			# tgt_sup_loader = torch.utils.data.DataLoader(
			#     target_train_dset, sampler=train_sampler, batch_size=self.args.batch_size,
			#     num_workers=0, drop_last=False
			# )
			# tgt_unsup_loader = torch.utils.data.DataLoader(
			#     target_train_dset, shuffle=True, batch_size=self.args.batch_size,
			#     num_workers=0, drop_last=False
			# )

			#以前までのやつ
			unsup_subset     = torch.utils.data.Subset(target_train_dset, unlabeled_indices)
			indexed_unsup    = unsup_subset  # ← ここだけでOK（サンプラは使わない）
			tgt_unsup_loader = torch.utils.data.DataLoader(
				indexed_unsup,
				batch_size=self.args.batch_size,
				shuffle=True,
				num_workers=0,
				drop_last=False
			)

			#今後の内容
			# if self.args.da_strat == "jumbot":
			# 	# ③ ターゲットの「確定ラベル」ルックアップを作成（未知は -1）
			# 	#    例：長さ = 全ターゲット数、labeled_indices だけ真のラベル、それ以外は -1
			# # ② 監督なしローダ：『全ターゲット』を回すように変更（←重要）
			# 	all_target_indices = self.train_idx  # = 全ターゲットの絶対インデックス
			# 	unsup_check_subset = torch.utils.data.Subset(target_train_dset, all_target_indices)
			# 	tgt_check_loader = torch.utils.data.DataLoader(
			# 		unsup_check_subset, batch_size=self.args.batch_size,
			# 		shuffle=True, num_workers=0, drop_last=False
			# 	)
			# 	target_label_lookup = torch.full((len(target_train_dset),), -1, dtype=torch.long)
			# 	if len(labeled_indices) > 0:
			# 		target_label_lookup[labeled_indices] = labels

			# 	opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.wd)

			# 	solver = get_solver(
			# 		self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_check_loader,
			# 		self.train_idx, opt_net_tgt, da_round, self.device, self.args
			# 	# ④ ルックアップを Solver に渡す（新規メソッド）
			# 	)
			# 	if hasattr(solver, "set_target_label_lookup"):
			# 		solver.set_target_label_lookup(target_label_lookup)


			# tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=0, \
			# 											   batch_size=self.args.batch_size, drop_last=False)
		# tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=0, \
		# 											   batch_size=self.args.batch_size, drop_last=False)			
		opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.wd)

		# Update discriminator adversarially with classifier
		lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)
		solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
							self.train_idx, opt_net_tgt, da_round, self.device, self.args)
		
		for epoch in range(self.args.adapt_num_epochs):
			if self.device.type == "cuda":
				torch.cuda.synchronize()
				time.sleep(2.0)  # 2秒休憩（値は調整）
			if self.args.da_strat == 'dann':
				opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=self.args.adapt_lr, \
										   betas=(0.9, 0.999), weight_decay=0)
				solver.solve(epoch, self.discriminator, opt_dis_adapt)
			elif self.args.da_strat in ['ft', 'mme','jumbot']:
				solver.solve(epoch)
			else:
				raise NotImplementedError
		
			lr_scheduler.step()
			# Save the current state of the model
		if self.args.da_strat == 'UOT' and hasattr(solver, "save_pi_final"):
			solver.save_pi_final(round_id=da_round)  # まとめて saved_pi_round{xx}.pt として保存

		return self.model

@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
	"""
	Uniform sampling 
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(RandomSampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.labels = dset.labels if dset.name == 'DomainNet' else dset.targets
		self.classes = np.unique(self.labels)
		self.dset = dset
		self.balanced = balanced

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)

@register_strategy('AADA')
class AADASampling(SamplingStrategy):
	"""
	Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(AADASampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.D = None
		self.E = None

	def query(self, n):
		"""
		s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
		"""
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=64, drop_last=False)

		# Get diversity and entropy
		all_log_probs, all_scores = [], []
		with torch.no_grad():
			for batch_idx, (data, target,_) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				scores = self.model(data)
				log_probs = nn.LogSoftmax(dim=1)(scores)
				all_scores.append(scores)
				all_log_probs.append(log_probs)

		all_scores = torch.cat(all_scores)
		all_log_probs = torch.cat(all_log_probs)

		all_probs = torch.exp(all_log_probs)
		disc_scores = nn.Softmax(dim=1)(self.discriminator(all_scores))
		# Compute diversity
		self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
		# Compute entropy
		self.E = -(all_probs*all_log_probs).sum(1)
		# scores = (self.D*self.E).sort(descending=True)[1]
		# 置き換え後（決定的：スコア降順の上位 n 件をそのまま採用）
		#データの状態確認用
    	# ★変更点：上位からそのまま n 件を採用
		sel_score_all = self.D*self.E
		score_sorted_idx = torch.argsort(sel_score_all, descending=True) # [N_u]
		q_idxs_local = score_sorted_idx[:n].cpu().numpy()                # ローカル(未ラベル配列内)の上位 n
		#修正前
		selected_global = idxs_unlabeled[q_idxs_local] 
		return selected_global

@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
	"""
	Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(BADGESampling, self).__init__(dset, train_idx, model, discriminator, device, args)

	def query(self, n):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=0, batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()

		if self.args.cnn == 'LeNet':
			emb_dim = 500
		elif self.args.cnn == 'ResNet34':
			emb_dim = 512

		tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])
		tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])
		tgt_lab = torch.zeros(len(data_loader.sampler))
		tgt_preds = torch.zeros(len(data_loader.sampler))
		batch_sz = self.args.batch_size
		
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				e1, e2 = self.model(data, with_emb=True)
				tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
				tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
				tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
				tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

		# Compute uncertainty gradient
		tgt_scores = nn.Softmax(dim=1)(tgt_emb)
		tgt_scores_delta = torch.zeros_like(tgt_scores)
		tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1
		
		# Uncertainty embedding
		badge_uncertainty = (tgt_scores-tgt_scores_delta)

		# Seed with maximum uncertainty example
		max_norm = utils.row_norms(badge_uncertainty.cpu().numpy()).argmax()

		_, q_idxs = utils.kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n, init=[max_norm])

		return idxs_unlabeled[q_idxs]

@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
	"""
	Implements CLUE: CLustering via Uncertainty-weighted Embeddings
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(CLUESampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.random_state = np.random.RandomState(1234)
		self.T = self.args.clue_softmax_t

	def query(self, n):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=0, \
												  batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()
		
		if self.args.cnn == 'LeNet':
			emb_dim = 500
		elif self.args.cnn == 'ResNet34':
			emb_dim = 512

		# Get embedding of target instances
		tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)		
		tgt_pen_emb = tgt_pen_emb.cpu().numpy()
		tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
		tgt_scores += 1e-8
		sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()
		
		# Run weighted K-means over embeddings
		km = KMeans(n)
		km.fit(tgt_pen_emb, sample_weight=sample_weights)
		
		# Find nearest neighbors to inferred centroids
		dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
		sort_idxs = dists.argsort(axis=1)
		q_idxs = []
		ax, rem = 0, n
		while rem > 0:
			q_idxs.extend(list(sort_idxs[:, ax][:rem]))
			q_idxs = list(set(q_idxs))
			rem = n-len(q_idxs)
			ax += 1

		return idxs_unlabeled[q_idxs]

@register_strategy('UOT')
class UOTSampling(SamplingStrategy):
    ...
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(UOTSampling, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.model = model

        # ラウンドごとに読むなら saved_pi_round{r}.pt を指定してください。ひとまず最新だけ使うなら：
        self.pi_list = torch.load("saved_pi_final.pt")  # [{"pi": [Ns,Nt], "idx_t_all": [Nt], ...}, ...]
        self.max_pi_xt = self._compute_max_pi_per_target(len(train_idx))
        self.D = None
        self.E = None

    def _compute_max_pi_per_target(self, num_target_samples):
        """
        各ターゲット（グローバル index）について、全ログから max pi を集約。
        """
        max_pi = torch.zeros(num_target_samples)
        for entry in self.pi_list:
            pi = entry["pi"].to(dtype=torch.float32).cpu()               # [Ns, Nt]
            idx_t_all = entry["idx_t_all"].to(torch.long).cpu() # [Nt]  ← 列順
            max_vals = pi.max(dim=0).values      # [Nt]  列（ターゲット）方向の最大
            # 列→グローバル index へ一括反映
            max_pi[idx_t_all] = torch.max(max_pi[idx_t_all], max_vals)
        return max_pi

    def query(self, n):
        """
        s(x_t) = (1 - max_{x_s in X_S} pi(x_s, x_t)) * H(G_y(G_f(x_t)))
        UOTベースの重みと分類不確かさ（エントロピー）の積をスコアとして選定
        """
        self.model.eval()

        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(
            self.dset, sampler=train_sampler, num_workers=0, batch_size=64, drop_last=False
        )

        all_log_probs, all_scores = [], []
        with torch.no_grad():
            for batch_idx, (data, target,_) in enumerate(data_loader):
                data = data.to(self.device)
                logits, gen_output = self.model(data, with_emb=True)
                prob = F.softmax(logits, dim=1)
                log_prob = F.log_softmax(logits, dim=1)

                all_scores.append(prob)
                all_log_probs.append(log_prob)

        all_scores = torch.cat(all_scores)
        all_log_probs = torch.cat(all_log_probs)
        all_probs = all_scores  # = softmax(logits)

        # エントロピーの計算
        self.E = -(all_probs * all_log_probs).sum(1)  # shape = [N_unlabeled]

        # UOT重み：1 - max pi
        weights = 1.0 - self.max_pi_xt[idxs_unlabeled]  # shape = [N_unlabeled]
        weights = weights.to(self.device)
        self.E = self.E.to(self.device)
        # スコア = 不確実性 × UOT重み
        scores = weights * self.E
        #scores = weights
        # 上位 n 件を選択
        top_idxs = torch.topk(scores, k=n, largest=True)[1]
        selected = idxs_unlabeled[top_idxs.cpu().numpy()]

        return selected

# sample.py の末尾付近に追加
@register_strategy('FORCED')
class ForcedSelection(SamplingStrategy):
    """
    事前に作った balanced_plan_runX.json に従って、各ラウンドの query() でその回のインデックスを返す。
    - args.forced_plan_dir に run*/balanced_plan_run{run_id}.json がある想定
    - run_id は 0,1,2... として、最初に見つかったファイルを使う（実運用では run を固定して回すのが簡単）
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(ForcedSelection, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.round_ctr = 1
        self.plan = {}

        plan_dir = getattr(args, "forced_plan_dir", "")
        if not plan_dir or not os.path.isdir(plan_dir):
            print("[FORCED] forced_plan_dir 未指定または存在しません。乱択にフォールバックします。")
            self.plan = {}
            return

        # run*/balanced_plan_runX.json から 1 つ選ぶ（基本は run0 を使う運用でOK）
        import glob, json
        cands = sorted(glob.glob(os.path.join(plan_dir, "run*", "balanced_plan_run*.json")))
        if not cands:
            print(f"[FORCED] 計画ファイルが見つかりません: {plan_dir}")
            self.plan = {}
            return

        use_path = cands[0]
        with open(use_path, "r", encoding="utf-8") as f:
            self.plan = json.load(f)
        print(f"[FORCED] using plan: {use_path}")

    def query(self, n):
        # このラウンドの予定
        key = str(self.round_ctr)
        planned = self.plan.get(key, [])
        self.round_ctr += 1

        # 未指定なら乱択
        if not planned:
            print("[FORCED] plan missing for this round -> random choice")
            return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)

        # “未ラベルプールのローカル添え字”を返す必要があるので、グローバル→ローカルに変換
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        unlabeled_global = self.train_idx[idxs_unlabeled]  # 全体の絶対インデックス配列

        # 計画の絶対インデックスを、unlabeled_global 内で探し、ローカル位置に変換
        planned = [int(x) for x in planned]
        mask = np.isin(unlabeled_global, np.array(planned, dtype=int))
        local_hits = np.where(mask)[0]

        if len(local_hits) >= n:
            # 予定数が多い場合は先頭 n 件（またはランダム n 件）を返す
            return idxs_unlabeled[np.random.choice(local_hits, size=n, replace=False)]
        else:
            # 予定が足りない分はランダム補完
            needed = n - len(local_hits)
            remaining_local = np.setdiff1d(np.arange(len(idxs_unlabeled)), local_hits, assume_unique=False)
            if len(remaining_local) > 0:
                add = np.random.choice(remaining_local, size=min(needed, len(remaining_local)), replace=False)
                local_all = np.concatenate([local_hits, add])
            else:
                local_all = local_hits
            return idxs_unlabeled[local_all]