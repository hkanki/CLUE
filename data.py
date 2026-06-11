import os
import random
import numpy as np
import h5py

import PIL
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import uot_utils
from uot_utils import BalancedBatchSampler, InfiniteSliceIterator

from pathlib import Path

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

def rgb_to_gray(img):
	return img.convert('L')

# data.py などに追加
import torch



class GaussianOutlierWrapper(torch.utils.data.Dataset):
    """
    - 外れ値にするサンプル集合も固定
    - ノイズも「サンプルごとに固定」（毎回同じ壊れ方）
    - base が Subset の場合でも安全（abs index で管理）
    """
    def __init__(self, base_dataset, outlier_ratio=0.1, sigma=0.2, seed=1234, clamp=(0.0, 1.0)):
        self.base = base_dataset
        self.n = len(base_dataset)
        self.sigma = float(sigma)
        self.clamp = clamp
        self.seed = int(seed)

        # Subset の場合：local i -> absolute idx の対応を持つ
        base_indices = getattr(base_dataset, "indices", None)
        if base_indices is None:
            self.abs_indices = np.arange(self.n, dtype=np.int64)
        else:
            self.abs_indices = np.array(base_indices, dtype=np.int64)

        # 外れ値（壊すサンプル）を absolute idx で固定選択
        rng = np.random.RandomState(self.seed)
        k = int(round(self.n * outlier_ratio))
        out_abs = rng.choice(self.abs_indices, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
        self.outlier_abs_set = set(map(int, out_abs))

        # 既存コード互換用（AL候補除外で参照されることがある）
        # “abs idx が取りうる最大値” がわからないケースがあるので、set と併用しつつ安全な形にする
        # 可能なら base がフルデータのときだけ bool 配列も持つ
        self.is_outlier_abs = None
        if base_indices is None:
            m = int(self.abs_indices.max()) + 1 if self.n > 0 else 0
            self.is_outlier_abs = np.zeros(m, dtype=bool)
            for a in self.outlier_abs_set:
                if 0 <= a < m:
                    self.is_outlier_abs[a] = True

        # ラベル配列をプロキシ（BalancedBatchSampler 等で必要になる場合）
        self.labels = getattr(base_dataset, "labels", None)
        self.targets = getattr(base_dataset, "targets", None)
        self.name = getattr(base_dataset, "name", "wrapped")
        
    # ★追加：DomainNet 用
    def get_num_classes(self):
        # base が持っていればそれを使う
        if hasattr(self.base, "get_num_classes"):
            return self.base.get_num_classes()
        # 無ければ labels/targets から推定
        if self.labels is not None:
            lab = np.array(self.labels[:len(self)], dtype=np.int64)
            return int(np.unique(lab).size)
        if self.targets is not None:
            t = np.array(self.targets, dtype=np.int64)
            return int(np.unique(t).size)
        raise AttributeError("get_num_classes: cannot infer num classes")
    
    def __len__(self):
        return self.n

    def _fixed_noise(self, x: torch.Tensor, abs_idx: int) -> torch.Tensor:
        """
        abs_idx から決定的にノイズを生成（順序に依存しない）
        """
        # サンプルごとの固定seed（衝突しにくいように大きめに混ぜる）
        sample_seed = (self.seed * 1000003 + int(abs_idx)) & 0x7fffffff

        # CPUで決定的に生成してから device に載せる（順序/workerの影響を受けにくい）
        g = torch.Generator(device="cpu")
        g.manual_seed(sample_seed)

        noise = torch.randn(x.shape, generator=g, dtype=x.dtype, device="cpu") * self.sigma
        return noise.to(x.device)

    def __getitem__(self, i):
        out = self.base[i]
        if len(out) == 3:
            x, y, idx = out
            # idx を信用せず、Subset対応のマップで abs idx を確定
            abs_idx = int(self.abs_indices[i])
        else:
            x, y = out
            abs_idx = int(self.abs_indices[i])
            idx = abs_idx

        if abs_idx in self.outlier_abs_set:
            x = x + self._fixed_noise(x, abs_idx)
            if self.clamp is not None:
                lo, hi = self.clamp
                x = torch.clamp(x, lo, hi)

        return x, y, idx



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

	

# class DomainNetDataset(torch.utils.data.Dataset):
# 	def __init__(self, name, domain, split, transforms):
# 		self.name = 'DomainNet'
# 		self.domain = domain
# 		self.split = split
# 		root = Path(r"H:\マイドライブ\study\aada+uot(office)_compleate\UOT+AADA(Ofice\CLUE\data")  # ← 実在する出力先
# 		self.file_path = root / f"{domain}_{split}.h5"        # 例: clipart_train.h5
# 		self.data, self.labels = None, None
# 		# with h5py.File(self.file_path, 'r') as file:
# 		# 	self.dataset_len = len(file["images"])
# 		# 	self.num_classes = len(set(list(np.array(file['labels']))))
# 		with h5py.File(self.file_path, 'r') as file:
# 			n_img = len(file['images'])
# 			n_lab = len(file['labels'])
# 			if n_img != n_lab:
# 				print(f"[WARN] {self.file_path.name}: images({n_img}) != labels({n_lab}) → min を使用")
# 				self.dataset_len = min(n_img, n_lab)   # ★ ここを修正
# 				self.num_classes = len(set(list(np.array(file['labels']))))		
# 		self.transforms = transforms
# 	def __len__(self):
# 		return self.dataset_len

# 	def __getitem__(self, idx):
# 		if self.data is None:
# 			self.data = h5py.File(self.file_path, 'r')["images"]
# 			self.labels = h5py.File(self.file_path, 'r')["labels"]
# 		datum, label = Image.fromarray(np.uint8(np.array(self.data[idx]))), np.array(self.labels[idx])
# 		return (self.transforms(datum), int(label), idx)

# 	def get_num_classes(self):
# 		# return self.num_classes
# 		#! Hardcoded
# 		return self.num_classes
class DomainNetDataset(torch.utils.data.Dataset):
    def __init__(self, name, domain, split, transforms):
        self.name = 'DomainNet'
        self.domain = domain
        self.split = split
        # root = Path(r"H:\マイドライブ\study\aada+uot(office)_compleate\UOT+AADA(Ofice\CLUE\data")
        root = Path(r"H:\マイドライブ\study\aada+uot(office)_compleate\UOT+AADA(Ofice\CLUE\data")
        self.file_path = root / f"{domain}_{split}.h5"

        # ★ H5 を1回だけ開いて保持
        self._h5 = h5py.File(self.file_path, 'r')
        self.data = self._h5['images']
        self.labels = self._h5['labels']

        n_img = len(self.data)
        n_lab = len(self.labels)
        if n_img != n_lab:
            print(f"[WARN] {self.file_path.name}: images({n_img}) != labels({n_lab}) → min を使用")

        # ★ 常に min を使って長さを確定（不一致でも一致でも）
        self.dataset_len = min(n_img, n_lab)

        # ★ クラス数も常に設定（必要分だけスライス）
        lab_np = np.array(self.labels[:self.dataset_len])
        self.num_classes = int(np.unique(lab_np).size)

        self.transforms = transforms

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # ★ 範囲チェック（デバッグに有用）
        assert 0 <= idx < self.dataset_len, f"idx {idx} out of [0,{self.dataset_len-1}]"
        img = Image.fromarray(np.uint8(np.array(self.data[idx])))
        label = int(self.labels[idx])
        return (self.transforms(img), label, idx)

    def get_num_classes(self):
        return self.num_classes

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except:
            pass

class ASDADataset:
	# Active Semi-supervised DA Dataset class
	def __init__(self, name, data_dir='data', valid_ratio=0.2, batch_size=128, augment=False):
		self.name = name
		self.data_dir = data_dir
		self.valid_ratio = valid_ratio
		self.batch_size = batch_size
		self.train_size = None
		self.train_dataset = None
		self.num_classes = None

	def get_num_classes(self):
		return self.num_classes

	def get_dsets(self, normalize=True, apply_transforms=True):
		if self.name == "mnist":
			mean, std = 0.5, 0.5
			normalize_transform = transforms.Normalize((mean,), (std,)) \
								  if normalize else transforms.Normalize((0,), (1,))
			train_transforms = transforms.Compose([
									   transforms.ToTensor(),
									   normalize_transform
								   ])
			test_transforms = transforms.Compose([
									   transforms.ToTensor(),
									   normalize_transform
									])

			train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=train_transforms)
			val_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=test_transforms)
			test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=test_transforms)
			train_dataset.name, val_dataset.name, test_dataset.name = 'DIGITS','DIGITS', 'DIGITS'
			self.num_classes = 10
		
		elif self.name == "svhn":
			mean, std = 0.5, 0.5
			normalize_transform = transforms.Normalize((mean,), (std,)) \
								  if normalize else transforms.Normalize((0,), (1,))
			#RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))
			train_transforms = transforms.Compose([
								   #RGB2Gray,
								   transforms.Lambda(rgb_to_gray),
								   transforms.Resize((28, 28)),
								   transforms.ToTensor(),
								   normalize_transform
							   ])
			test_transforms = transforms.Compose([
								   #RGB2Gray,
								   transforms.Lambda(rgb_to_gray),
								   transforms.Resize((28, 28)),
								   transforms.ToTensor(),
								   normalize_transform
							   ])

			train_dataset = datasets.SVHN(self.data_dir, split='train', download=True, transform=train_transforms)
			val_dataset = datasets.SVHN(self.data_dir, split='train', download=True, transform=test_transforms)
			test_dataset = datasets.SVHN(self.data_dir, split='test', download=True, transform=test_transforms)
			self.num_classes = 10

		elif self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart", "amazon","dslr","webcam","art","realworld","product"]:

			normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
								  if normalize else transforms.Normalize([0, 0, 0], [1, 1, 1])
			
			if apply_transforms:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(256),
						transforms.RandomCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						normalize_transform
					]),
				}
			else:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(224),
						transforms.ToTensor(),
						normalize_transform
					]),
				}

			data_transforms['test'] = transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					normalize_transform
				])

			train_dataset = DomainNetDataset('DomainNet', self.name, 'train', data_transforms['train'])
			val_dataset = DomainNetDataset('DomainNet', self.name, 'val', data_transforms['test'])
			test_dataset = DomainNetDataset('DomainNet', self.name, 'test', data_transforms['test'])
			print(self.name, "num_classes:", train_dataset.get_num_classes())
			self.num_classes = train_dataset.get_num_classes()

		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset

		return train_dataset, val_dataset, test_dataset

	# def get_loaders(self, shuffle=True, num_workers=0, normalize=True):
	# 	if not self.train_dataset: self.get_dsets(normalize=normalize)
		
	# 	num_train = len(self.train_dataset)
	# 	self.train_size = num_train

	# 	if self.name in ["mnist", "svhn"]:
			
	# 		indices = list(range(num_train))
	# 		split = int(np.floor(self.valid_ratio * num_train))
	# 		if shuffle == True: np.random.shuffle(indices)
	# 		train_idx, valid_idx = indices[split:], indices[:split]
			
	# 		train_sampler = SubsetRandomSampler(train_idx)
	# 		valid_sampler = SubsetRandomSampler(valid_idx)

	# 	elif self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart"]:

	# 		train_idx = np.arange(len(self.train_dataset))
	# 		train_sampler = SubsetRandomSampler(train_idx)
	# 		valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))

	# 	train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
	# 											   batch_size=self.batch_size, num_workers=num_workers)
	# 	val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, batch_size=self.batch_size)
	# 	test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

	# 	return train_loader, val_loader, test_loader, train_idx

	def get_loaders(self, shuffle=True, num_workers=0, normalize=True, use_balanced_sampler=False,use_balance = False):
		if not self.train_dataset:
			self.get_dsets(normalize=normalize)

		num_train = len(self.train_dataset)
		self.train_size = num_train

		if self.name in ["mnist", "svhn"]:
			indices = list(range(num_train))
			split = int(np.floor(self.valid_ratio * num_train))
			if shuffle:
				np.random.shuffle(indices)
			train_idx, valid_idx = indices[split:], indices[:split]

			if use_balanced_sampler:
				# ラベルを取得：train_idxを使ってサブセットを作る
				train_dataset_subset = torch.utils.data.Subset(self.train_dataset, train_idx)
				
				# 元の train_dataset からインデックスに基づいてラベルを抽出
				labels = torch.tensor([train_dataset_subset[i][1] for i in range(len(train_dataset_subset))])

				if use_balance:
					indexed_dataset = DatasetWithIndex(train_dataset_subset)
					train_loader = torch.utils.data.DataLoader(
						indexed_dataset,
						batch_sampler=BalancedBatchSampler(labels, self.batch_size),
						num_workers=num_workers
					)
				else:
					# 元データ基準のインデックス train_idx をそのまま使えるよう
					# DatasetWithIndex(self.train_dataset) + SubsetRandomSampler(train_idx)
					indexed_dataset = DatasetWithIndex(self.train_dataset)
					train_sampler = SubsetRandomSampler(train_idx)
					train_loader = torch.utils.data.DataLoader(
						indexed_dataset,
						sampler=train_sampler,
						batch_size=self.batch_size,   # ← 忘れずに
						num_workers=num_workers
					)
			else:
				train_sampler = SubsetRandomSampler(train_idx)
				train_loader = torch.utils.data.DataLoader(
					self.train_dataset,
					sampler=train_sampler,
					batch_size=self.batch_size,
					num_workers=num_workers
				)

			valid_sampler = SubsetRandomSampler(valid_idx)
			val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, batch_size=self.batch_size)

		elif self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart", "amazon","dslr","webcam","art","realworld","product"]:
			train_idx = np.arange(len(self.train_dataset))
			if use_balanced_sampler:
				# ★ ここを書き換え
				if isinstance(self.train_dataset, DomainNetDataset):
					n = len(self.train_dataset)
					labels = torch.from_numpy(np.array(self.train_dataset.labels[:n], dtype=np.int64))
				else:
					labels = torch.tensor([self.train_dataset[i][1] for i in range(len(self.train_dataset))])

				train_loader = torch.utils.data.DataLoader(
					self.train_dataset,
					batch_sampler=BalancedBatchSampler(labels, self.batch_size),
					num_workers=num_workers
				)
			else:
				train_sampler = SubsetRandomSampler(train_idx)
				train_loader = torch.utils.data.DataLoader(
					self.train_dataset,
					sampler=train_sampler,
					batch_size=self.batch_size,
					num_workers=num_workers
				)

			valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))
			val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, batch_size=self.batch_size)
		#UOT用でやっている
		test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

		return train_loader, val_loader, test_loader, train_idx