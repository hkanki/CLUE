# -*- coding: utf-8 -*-
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ot

from .solver import register_solver
sys.path.append('../../')
import utils

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

class BaseSolver:
	"""
	Base DA solver class
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		self.net = net
		self.src_loader = src_loader
		self.tgt_sup_loader = tgt_sup_loader
		self.tgt_unsup_loader = tgt_unsup_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.da_round = da_round
		self.device = device
		self.args = args

	def solve(self, epoch):
		pass

@register_solver('ft')
class TargetFTSolver(BaseSolver):
	"""
	Finetune on target labels
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(TargetFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch):
		"""
		Finetune on target labels
		"""		
		self.net.train()		
		if (self.da_round > 0): tgt_sup_iter = iter(self.tgt_sup_loader)
		info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
		while True:
			try:
				data_t, target_t = next(tgt_sup_iter)
				data_t, target_t = data_t.to(self.device), target_t.to(self.device)
			except: break
			
			self.tgt_opt.zero_grad()
			output = self.net(data_t)
			loss = nn.CrossEntropyLoss()(output, target_t)
			info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
			info_str += ' Target Sup. Loss: {:.3f}'.format(loss.item())
			
			loss.backward()
			self.tgt_opt.step()
		
		if epoch % 10 == 0: print(info_str)

# @register_solver('jumbot')
# class JUMBOTSolver(BaseSolver):
# 	"""
# 	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
# 	"""
# 	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
# 		super(JUMBOTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
# 		# self.target_test_loader = target_test_loader

# 	def save_pi_final(self, round_id=None):
#         # 常に round 付きも保存
# 		if round_id is not None:
# 			torch.save(self.saved_pi, f"saved_pi_round{round_id:02d}.pt")
#         # 直近を "final" にも上書き保存（読み側は常にこれを読めばOK）
# 		torch.save(self.saved_pi, "saved_pi_final.pt")
# 	def solve(self, epoch):
# 		eta1 = 0.001
# 		eta2 = 0.01
# 		tau = 0.1
# 		epsilon = 0.01
# 		if(epoch==0):
# 			print(f"eta1: {eta1}, eta2: {eta2}, epsilon: {epsilon}, tau: {tau}")
# 		"""
# 		Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
# 									ent. minimization on target + DANN on source<->target
# 		"""
# 		gan_criterion = nn.CrossEntropyLoss()

# 		self.net.train()
# 		self.saved_pi = []  # 外に保存
		
# 		if self.da_round == 0:
# 			pass
# 		else:
# 			tgt_sup_iter = iter(self.tgt_sup_loader)

# 		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)		
# 		for batch_idx, ((data_s, label_s,_), (data_tu, label_tu,idx_t)) in enumerate(joint_loader):
# 			data_s = data_s.to(self.device)
# 			label_s = label_s.to(self.device)
# 			data_tu = data_tu.to(self.device)
# 			idx_t = idx_t.to(self.device)

# 			if self.da_round > 0:
# 				try:
# 					data_ts, label_ts ,_= next(tgt_sup_iter)
# 					data_ts = data_ts.to(self.device)
# 					label_ts = label_ts.to(self.device)
				
# 				except: 
# 					break

# 			# Train with target labels
# 			score_s, emb_s = self.net(data_s,with_emb=True)
# 			xeloss_src = nn.CrossEntropyLoss()(score_s, label_s)

# 			info_str = "[Train DANN] Epoch: {}".format(epoch)
# 			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
# 			sup_loss = 0.0

# 			if self.da_round > 0:
# 				score_ts, emb_ts = self.net(data_ts,with_emb=True)
# 				ts_loss = F.cross_entropy(score_ts, label_ts)
# 				sup_loss = ts_loss

# 			# extract and concat features
# 			score_tu, emb_tu = self.net(data_tu,with_emb=True)

# 			embed_cost = torch.cdist(emb_s, emb_tu, p=2).pow(2)
# 			num_classes = score_s.size(1)
#                 # ys のラベルをワンホットエンコーディング
# 			ys = F.one_hot(label_s, num_classes=num_classes).float()
#                 #t_cost：sourceのラベルとtargetの予測確率の交差エントロピー（Sinkhorn形式）
			
# 			pred_xt = F.softmax(score_tu, dim=1)
# 			#pred_xt = torch.clamp(pred_xt, min=1e-6, max=1.0)

# 			t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
#                 #total_cost：これらを重み付きで合成
# 			#ミニバッチのコスト行列C_{I,J}
# 			total_cost = eta1 * embed_cost + eta2 * t_cost
# 			#total_cost = torch.clamp(total_cost, min=0, max=1e6)

# 			if torch.isnan(total_cost).any() or torch.isinf(total_cost).any():
# 				print('[JUMBOT] total_cost is NaN/Inf -> abort this hyperparam set')
# 				return False				# エラー処理やスキップ

#                 #OT computation
#                 # a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
#                 # OT: stay on GPU for speed, convert to numpy only at final step
#                 #a, b: ソース・ターゲットサンプルの一様分布
#                 # a: ソースミニバッチのサンプルごとに 1 𝑁 N 1  の質量を持つ一様分布（長さ N = B_s のベクトル） 
#                 # b: ターゲットミニバッチのサンプルごとに 1 𝑀 M 1  の質量を持つ一様分布（長さ M = B_t のベクトル）
# 			#u_m(一様分布ベクトル)
# 			a = torch.full((emb_s.size(0),), 1.0 / emb_s.size(0), device=self.device)
# 			b = torch.full((emb_tu.size(0),), 1.0 / emb_tu.size(0), device=self.device)
# 			# a = a / a.sum()
# 			# b = b / b.sum()   
#                 #                                              self.epsilon, self.tau)
#                 #Sinkhorn-Knopp を使い、GPU→CPUで計算 → 再度Tensor化して使用
# 			#batchつけていなかった
# 			try:
# 				#h(u_m,u_m,C_{I,J})(左の式におけるOT計算)
# 				pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
# 					a.detach().cpu().numpy(),
# 					b.detach().cpu().numpy(),
# 					total_cost.detach().cpu().numpy(),
# 					epsilon,
# 					tau
# 				)
# 			except Exception as e:
# 				print(f'[JUMBOT] Sinkhorn failed: {e} -> abort this hyperparam set')
# 				return False

# 			if not np.isfinite(pi_np).all():
# 				print('[JUMBOT] pi contains NaN/Inf -> abort this hyperparam set')
# 				return False


# 			pi = torch.tensor(pi_np, device=self.device)
			
# 			self.saved_pi.append({
#                     "pi":        pi.detach().cpu().to(torch.float32),      # [Ns, Nt]
#                     "idx_t_all": idx_t.detach().cpu().to(torch.long)    # [Nt]  ← 列順そのまま
#             })

# 			self.tgt_opt.zero_grad()

#             #da_loss: ドメイン間の分布のズレを埋めるための OT 損失(h(u_m,u_m,C_{I,J})
# 			#論文内の(6)に当たる計算
# 			da_loss = torch.sum(pi * total_cost)
# 			# compute loss for disciminator
# 			# --- da_loss 監視：0（実質0含む）や非有限ならスキップ ---
# 			da_val = da_loss.detach().item()
# 			if (not np.isfinite(da_val)) or (da_val <= 1e-12):
# 				print(f'[JUMBOT] da_loss={da_val:.3e} -> abort this hyperparam set')
# 				return False
# 			loss_final = da_loss+ (sup_loss if isinstance(sup_loss, torch.Tensor) else 0.0)

# 			loss_final.backward()

# 			self.tgt_opt.step()
		
# 			# log net update info
# 			info_str += " da_loss loss: {:.3f}".format(da_loss.item())
# 			info_str += " embed_cost mean: {:.3f}".format(embed_cost.mean().item())
# 			info_str += " t_cost mean: {:.3f}".format(t_cost.mean().item())
   
# 			# if epoch % 1 == 0:  # 毎エポック
# 			# 	acc = utils.model_eval(self.net, self.target_test_loader, device=self.device)
# 			# 	print(f"[Eval] Epoch {epoch}: Target Accuracy = {acc:.2f}%")

# 		if epoch%10 == 0: print(info_str)
# 		return True
@register_solver('jumbot')
class JUMBOTSolver(BaseSolver):
	"""
	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(JUMBOTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	# def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args, eta1, eta2, epsilon, tau):
	#   super(JUMBOTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
		# self.target_test_loader = target_test_loader
		# self.eta1 = 0.5
		self.eta1 = 0.001
		self.eta2 = 0.01
		self.epsilon = 0.02
		self.tau = 0.5

	def save_pi_final(self, round_id=None):
        # 常に round 付きも保存
		if round_id is not None:
			torch.save(self.saved_pi, f"saved_pi_round{round_id:02d}.pt")
        # 直近を "final" にも上書き保存（読み側は常にこれを読めばOK）
		torch.save(self.saved_pi, "saved_pi_final.pt")
	def solve(self, epoch):
		eta1 = self.eta1
		eta2 = self.eta2
		epsilon = self.epsilon
		tau = self.tau
		
		if(epoch==0):
			print(f"eta1: {eta1}, eta2: {eta2}, epsilon: {epsilon}, tau: {tau}")
		"""
		Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
									ent. minimization on target + DANN on source<->target
		"""
		gan_criterion = nn.CrossEntropyLoss()

		self.net.train()
		self.saved_pi = []  # 外に保存
		
		if self.da_round == 0:
			pass
		else:
			tgt_sup_iter = iter(self.tgt_sup_loader)

		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)		
		for batch_idx, ((data_s, label_s,_), (data_tu, label_tu,idx_t)) in enumerate(joint_loader):
			data_s = data_s.to(self.device)
			label_s = label_s.to(self.device)
			data_tu = data_tu.to(self.device)
			idx_t = idx_t.to(self.device)

			if self.da_round > 0:
				try:
					data_ts, label_ts ,_= next(tgt_sup_iter)
					data_ts = data_ts.to(self.device)
					label_ts = label_ts.to(self.device)
				
				except: 
					break

			# Train with target labels
			score_s, emb_s = self.net(data_s,with_emb=True)
			xeloss_src = nn.CrossEntropyLoss()(score_s, label_s)
			# 追加: Src Sup loss が大きすぎたらこのハイパラをスキップ
			THRESH_SRC_LOSS = 1.0
			x_val = float(xeloss_src.detach().item())
			if (not np.isfinite(x_val)) or (x_val > THRESH_SRC_LOSS):
				print(f'[JUMBOT] Src Sup loss={x_val:.3f} (>{THRESH_SRC_LOSS}) -> abort this hyperparam set')
				return False
			info_str = "[Train DANN] Epoch: {}".format(epoch)
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
			sup_loss = 0.0

			if self.da_round > 0:
				score_ts, emb_ts = self.net(data_ts,with_emb=True)
				ts_loss = F.cross_entropy(score_ts, label_ts)
				sup_loss = ts_loss

			# extract and concat features
			score_tu, emb_tu = self.net(data_tu,with_emb=True)
			emb_s = F.normalize(emb_s, dim=1)
			emb_tu = F.normalize(emb_tu, dim=1)
			embed_cost = torch.cdist(emb_s, emb_tu, p=2).pow(2)
                # ys のラベルをワンホットエンコーディング
			num_classes = score_s.size(1)
			ys = F.one_hot(label_s, num_classes=num_classes).float()
                #t_cost：sourceのラベルとtargetの予測確率の交差エントロピー（Sinkhorn形式）
			
			pred_xt = F.softmax(score_tu, dim=1)
			pred_xt = torch.clamp(pred_xt, min=1e-6, max=1.0)

			t_cost = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
                #total_cost：これらを重み付きで合成
			total_cost = eta1 * embed_cost + eta2 * t_cost
			total_cost = total_cost / (total_cost.median() + 1e-12)
			#total_cost = torch.clamp(total_cost, min=0, max=1e6)
			# εとτをコストに合わせて自動調整
			ratio = (total_cost.max().detach().item()) / max(epsilon, 1e-12)
			if ratio > 80.0:
				epsilon = total_cost.max().detach().item() / 80.0
			tau = max(min(tau, 0.3 * epsilon), 1e-3)  # ψを抑える

			if torch.isnan(total_cost).any() or torch.isinf(total_cost).any():
				print('[JUMBOT] total_cost is NaN/Inf -> abort this hyperparam set')
				return False				# エラー処理やスキップ

                #OT computation
                # a, b = ot.unif(g_xs_mb.size()[0]), ot.unif(g_xt_mb.size()[0])
                # OT: stay on GPU for speed, convert to numpy only at final step
                #a, b: ソース・ターゲットサンプルの一様分布
                # a: ソースミニバッチのサンプルごとに 1 𝑁 N 1  の質量を持つ一様分布（長さ N = B_s のベクトル） 
                # b: ターゲットミニバッチのサンプルごとに 1 𝑀 M 1  の質量を持つ一様分布（長さ M = B_t のベクトル）
			
			# a = torch.full((emb_s.size(0),), 1.0 / emb_s.size(0), device=self.device)
			# b = torch.full((emb_tu.size(0),), 1.0 / emb_tu.size(0), device=self.device)
			a_np = torch.full((emb_s.size(0),), 1.0/emb_s.size(0), device=self.device, dtype=torch.float64).cpu().numpy()
			b_np = torch.full((emb_tu.size(0),), 1.0/emb_tu.size(0), device=self.device, dtype=torch.float64).cpu().numpy()
			C_np = total_cost.detach().double().cpu().numpy()			
			# a = a / a.sum()
			# b = b / b.sum()   
                #                                              self.epsilon, self.tau)
                #Sinkhorn-Knopp を使い、GPU→CPUで計算 → 再度Tensor化して使用
			#batchつけていなかった
			try:
				pi_np = ot.unbalanced.sinkhorn_knopp_unbalanced(
					a_np,
					b_np,
					C_np,
					epsilon,
					tau,
					stopThr=1e-6, numItermax=500
				)
			except Exception as e:
				print(f'[JUMBOT] Sinkhorn failed: {e} -> abort this hyperparam set')
				return False

			if not np.isfinite(pi_np).all():
				print('[JUMBOT] pi contains NaN/Inf -> abort this hyperparam set')
				return False



			pi = torch.tensor(pi_np, device=self.device)
			if (not torch.isfinite(pi).all()) or (pi.sum() <= 1e-12):
				print(f'[JUMBOT] invalid pi (sum={pi.sum().item():.2e}) -> abort this hyperparam set')
				return False
			self.saved_pi.append({
                    "pi":        pi.detach().cpu().to(torch.float32),      # [Ns, Nt]
                    "idx_t_all": idx_t.detach().cpu().to(torch.long)    # [Nt]  ← 列順そのまま
            })

			self.tgt_opt.zero_grad()

            #da_loss: ドメイン間の分布のズレを埋めるための OT 損失
			da_loss =  torch.sum(pi * total_cost)
			# compute loss for disciminator
			# --- da_loss 監視：0（実質0含む）や非有限ならスキップ ---
			da_val = da_loss.detach().item()
			if (not np.isfinite(da_val)) or (da_val <= 1e-12):
				print(f'[JUMBOT] da_loss={da_val:.3e} -> abort this hyperparam set')
				return False
			lambda_da = 0.25
			loss_final = xeloss_src + lambda_da*da_loss+ (sup_loss if isinstance(sup_loss, torch.Tensor) else 0.0)

			loss_final.backward()

			self.tgt_opt.step()
		
			# log net update info
			info_str += " da_loss loss: {:.3f}".format(da_loss.item())
			info_str += " embed_cost mean: {:.3f}".format(embed_cost.mean().item())
			info_str += " t_cost mean: {:.3f}".format(t_cost.mean().item())
   
			# if epoch % 1 == 0:  # 毎エポック
			# 	acc = utils.model_eval(self.net, self.target_test_loader, device=self.device)
			# 	print(f"[Eval] Epoch {epoch}: Target Accuracy = {acc:.2f}%")

		if epoch%4 == 0: print(info_str)
		return True

@register_solver('dann')
class DANNSolver(BaseSolver):
	"""
	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(DANNSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch, disc, disc_opt):
		"""
		Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
									ent. minimization on target + DANN on source<->target
		"""
		gan_criterion = nn.CrossEntropyLoss()
		cent = utils.ConditionalEntropyLoss().to(self.device)

		self.net.train()
		disc.train()
		
		if self.da_round == 0:
			src_sup_wt, lambda_unsup, lambda_cent = 1.0, 0.01, 0.01 # Hardcoded for unsupervised DA
		else:
			src_sup_wt, lambda_unsup, lambda_cent = self.args.src_sup_wt, self.args.unsup_wt, self.args.cent_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)

		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)		
		for batch_idx, ((data_s, label_s,_), (data_tu, label_tu,_)) in enumerate(joint_loader):
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)

			if self.da_round > 0:
				try:
					data_ts, label_ts,_ = next(tgt_sup_iter)
					data_ts = data_ts.to(self.device)
					label_ts = label_ts.to(self.device)
				except: break

			# zero gradients for optimizers
			self.tgt_opt.zero_grad()
			disc_opt.zero_grad()

			# Train with target labels
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt*nn.CrossEntropyLoss()(score_s, label_s)

			info_str = "[Train DANN] Epoch: {}".format(epoch)
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())                    

			xeloss_tgt = 0
			if self.da_round > 0:
				score_ts = self.net(data_ts)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
				info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			# extract and concat features
			score_tu = self.net(data_tu)
			f = torch.cat((score_s, score_tu), 0)

			# predict with discriminator
			f_rev = utils.ReverseLayerF.apply(f)
			pred_concat = disc(f_rev)

			target_dom_s = torch.ones(len(data_s)).long().to(self.device)
			target_dom_t = torch.zeros(len(data_tu)).long().to(self.device)
			label_concat = torch.cat((target_dom_s, target_dom_t), 0)

			# compute loss for disciminator
			loss_domain = gan_criterion(pred_concat, label_concat)
			loss_cent = cent(score_tu)

			loss_final = (xeloss_src + xeloss_tgt) + (lambda_unsup * loss_domain) + (lambda_cent * loss_cent)

			loss_final.backward()

			self.tgt_opt.step()
			disc_opt.step()
		
			# log net update info
			info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())		
			info_str += " Ent Loss: {:.3f}".format(lambda_cent * loss_cent.item())		
		
		if epoch%10 == 0: print(info_str)

@register_solver('mme')
class MMESolver(BaseSolver):
	"""
	Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(MMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch):
		"""
		Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
										adversarial ent. minimization on unlabeled target
		"""
		self.net.train()		
		src_sup_wt, lambda_adent = self.args.src_sup_wt, self.args.unsup_wt

		if self.da_round == 0:
			src_sup_wt, lambda_unsup = 1.0, 0.1
		else:
			src_sup_wt, lambda_unsup = self.args.src_sup_wt, self.args.unsup_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)


		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
		for batch_idx, ((data_s, label_s), (data_tu, label_tu)) in enumerate(joint_loader):			
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)
			
			if self.da_round > 0:
				try:
					data_ts, label_ts = next(tgt_sup_iter)
					data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
				except: break

			# zero gradients for optimizer
			self.tgt_opt.zero_grad()
					
			# log basic adapt train info
			info_str = "[Train Minimax Entropy] Epoch: {}".format(epoch)

			# extract features
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)
			
			# log discriminator update info
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
			
			xeloss_tgt = 0
			if self.da_round > 0:
				score_ts = self.net(data_ts)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
				info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			xeloss = xeloss_src + xeloss_tgt
			xeloss.backward()
			self.tgt_opt.step()

			# Add adversarial entropy
			self.tgt_opt.zero_grad()

			score_tu = self.net(data_tu, reverse_grad=True)
			probs_tu = F.softmax(score_tu, dim=1)
			loss_adent = lambda_adent * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
			loss_adent.backward()
			
			self.tgt_opt.step()
			
			# Log net update info
			info_str += " MME loss: {:.3f}".format(loss_adent.item())		
		
		if epoch%10 == 0: print(info_str)