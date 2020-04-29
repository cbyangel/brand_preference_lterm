import pandas as pd
import numpy as np
import pickle as pk

import os
import sys
import time

import xlearn as xl
import databricks.koalas as ks
from pathlib import Path
import subprocess

class FModel:

	def __init__(self, path=None, sc=None):
		self.MY_HOME = "/home/brand_lterm_preference"
		self.OPT_HOME = "/opt/download/users/22934"
		self.brand_feat_dict = {'brd_id':[]}
		self.col_list = None
		self.feature_idx_dict = {}

		self.pcid_feature_dict = {}
		self.pcid_lcate_dict = {}
		self.brand_feature_dict = {}
		self.brand_lcate_dict = {}

		self.train = None
		self.validate = None
		self.test = None

		if path is not None:
			self.load(path, sc)



	def df_to_table(self, df, tbl_name):
		if df is not None and tbl_name is not None:
			result_df = ks.from_pandas(df)
			result_df.to_table(name=tbl_name, format='orc', mode='overwrite')



	def make_embedding(self, df, target):
		feat_dict = {target : []}
		for l_cate in self.col_list:
			feat_dict[str(l_cate)] = []

		idx_map = {}
		idx = 0
		for row in df.itertuples():
			target_id = getattr(row, target)
			if target_id not in idx_map:
				idx_map.setdefault(target_id, idx)
				idx = idx + 1
				feat_dict[target].append(target_id)
				for l_cate in self.col_list:
					feat_dict[str(l_cate)].append(0)

			pos_idx = idx_map[target_id]
			feat_dict[str(row.cate1)][pos_idx] = row.view_cnt

		feat_df = pd.DataFrame(feat_dict)
		feat_df['tot'] = feat_df[[str(x) for x in self.col_list]].sum(axis=1)
		for col in [str(x) for x in self.col_list]:
			feat_df[col] = np.round(feat_df[col] / feat_df['tot'], 2)

		return feat_df


	def make_processing(self, df, target):
		for row in df.itertuples():
			emb = []
			lcate_pos = []
			for idx in range(2, len(self.col_list)):
				if row[idx] > 0:
					if target == 'pcid':
						emb.append(str(idx - 2) + ":" + str(row[idx]))
					elif target == 'brd_id':
						emb.append(str(idx - 2 + self.feature_idx_dict['brand_emb']) + ":" + str(row[idx]))

					if row[idx] >= 0.1:
						lcate_pos.append(idx)

			if len(emb) > 0:
				if target == 'pcid':
					self.pcid_feature_dict[str(row.pcid)] = " ".join(emb)
					self.pcid_lcate_dict[str(row.pcid)] = lcate_pos
				elif target == 'brd_id':
					self.brand_feature_dict[str(row.brd_id)] = " ".join(emb)
					self.brand_lcate_dict[str(row.brd_id)] = lcate_pos




	def make_recall(self):
		df = ks.sql(Path('./brand_lterm_preference/get_recall.sql').read_text()).to_pandas()
		brd_df = ks.sql(Path('./brand_lterm_preference/get_brand.sql').read_text()).to_pandas()
		merge_df = df.merge(brd_df, on='brand', how='left')

		# preprocessing
		merge_df = merge_df.dropna()
		merge_df['cnt'] = merge_df['cnt'].astype(int).astype(str)
		merge_df['score'] = merge_df['score'].astype(str)
		merge_df = merge_df[['userid', 'brand', 'score', 'brd_nm', 'cnt']]

		final_df = merge_df.groupby('userid').agg(lambda x: '^'.join(list(x)) if x.name == 'brd_nm' else ','.join(list(x))).reset_index()
		final_df['prdid'] = ''
		self.df_to_table(final_df, 'members_bycho.brand_lterm_preference')



	def make_brand_index(self):
		large_cate_df = ks.sql(Path('./brand_lterm_preference/get_large_cate.sql').read_text()).to_pandas()
		self.col_list = large_cate_df.cate1.unique().tolist()
		for cate1 in self.col_list:
			self.brand_feat_dict[cate1] = []

		self.feature_idx_dict = {'pcid_emb':0, 'brand_emb':len(self.col_list), 'ismatch':len(self.col_list)*2 + 1}
		tmp_df = pd.DataFrame([self.feature_idx_dict])
		self.df_to_table(tmp_df, "members_bycho.brand_feature_idx")



	def make_pcid_feat(self):
		pcid_emb_df = ks.sql(Path('./brand_lterm_preference/get_pcid_bhv.sql').read_text()).to_pandas()
		pcid_emb_df = pcid_emb_df.dropna()
		pcid_emb_df['cate1'] = pcid_emb_df['cate1'].astype(int)

		pcid_feat_df = self.make_embedding(pcid_emb_df, 'pcid')
		self.make_processing(pcid_feat_df, 'pcid')

		print(self.pcid_feature_dict['mEVEBGZUD8KY93ykgHWIw'])

		tmp = [[pcid, feat, self.pcid_lcate_dict[pcid]]
		        for pcid, feat in self.pcid_feature_dict.items()
		        ]
		tmp_df = pd.DataFrame(tmp, columns=['pcid', 'feat1', 'feat2'])
		self.df_to_table(tmp_df, "members_bycho.pcid_features")



	def make_brand_feat(self):
		brand_emb_df = ks.sql(Path('./brand_lterm_preference/get_brand_bhv.sql').read_text()).to_pandas()
		brand_emb_df = brand_emb_df.dropna()
		brand_emb_df['cate1'] = brand_emb_df['cate1'].astype(int)

		brand_feat_df = self.make_embedding(brand_emb_df, 'brd_id')
		self.make_processing(brand_feat_df, 'brd_id')

		print(self.brand_feature_dict['120402'])
		print(self.brand_lcate_dict['120402'])

		tmp = [[brd_id, feat, self.brand_lcate_dict[brd_id]]
		       for brd_id, feat in self.brand_feature_dict.items()
		       ]
		tmp_df = pd.DataFrame(tmp, columns=['brd_id', 'feat1', 'feat2'])
		self.df_to_table(tmp_df, "members_bycho.brand_features")




	def make_train_validate_testset(self):
		dataset_df = ks.sql(Path('./brand_lterm_preference/get_dataset.sql').read_text()).to_pandas()
		self.train, self.validate, self.test = np.split(dataset_df.sample(frac=1), [int(.6 * len(dataset_df)), int(.8 * len(dataset_df))])
		filename_list = ['train.txt', 'validate.txt', 'test.txt']
		df_list = [self.train, self.validate, self.test]

		try:
			for df, filename in zip(df_list, filename_list):
				txt_file = open(filename, 'w')
				for row in df.itertuples():
					vec = []
					label = row.istarget
					if (str(row.pcid) in self.pcid_feature_dict) & (str(row.brand) in self.brand_feature_dict):
						ismatch = len(set(self.pcid_lcate_dict[str(row.pcid)]) & set(self.brand_lcate_dict[str(row.brand)]))
						pcid_feat = self.pcid_feature_dict[str(row.pcid)]
						brand_feat = self.brand_feature_dict[str(row.brand)]
						vec.append(str(label))
						vec.append(pcid_feat)
						vec.append(brand_feat)
						if ismatch > 0:
							vec.append(str(self.feature_idx_dict['ismatch']) + ":" + str(1))
						txt_file.write("%s\n" % " ".join(vec))
				txt_file.close()
		except IOError as e:
			print("I/O error({0}) : {1}]".format(e.errno, e.strerror))
		else:
			self.move_to_opt(filename_list)




	def move_to_opt(self, files):
		return_val = None
		try:
			if type(files) == str:
				cmd = " ".join(['cp', files, self.OPT_HOME])
				process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
				return_val = process.wait()
			else:
				for f in files:
					cmd = " ".join(['cp', f, self.OPT_HOME])
					process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
					return_val = process.wait()
		except OSError:
			print('fail to run command')
		else:
			return return_val



	def run_model(self):
		return_val = None
		try:
			predict_file = "/".join([self.MY_HOME, 'xlearn/build/xlearn_predict'])
			test_file = "/".join([self.OPT_HOME, 'sterm.txt'])
			model_file = "/".join([self.OPT_HOME, 'model.out'])
			cmd = " ".join([predict_file, test_file, model_file])
			process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
			return_val = process.wait()
		except OSError:
			print('fail to run command')
		else:
			return return_val




	def create_model(self):
		print("model start")

		fm_model = xl.create_fm()

		train_file = "/".join([self.OPT_HOME, 'train.txt'])
		validate_file = "/".join([self.OPT_HOME, 'validate.txt'])
		test_file = "/".join([self.OPT_HOME, 'test.txt'])
		param = {'task': 'binary', 'epoch': 10, 'lr': 0.2, 'lambda': 0.002, 'metric': 'auc'}

		fm_model.setTrain(train_file)
		fm_model.setValidate(validate_file)

		fm_model.fit(param, 'model.out')
		fm_model.setTest(test_file)
		fm_model.setSigmoid()
		fm_model.predict('model.out', "output.txt")

		model_file = "/".join([self.MY_HOME, 'model.out'])
		print(model_file)
		self.move_to_opt(model_file)




	def sterm_reranking(self):
		print("start reranking!!")
		brand_feat_df = ks.sql(Path('./brand_lterm_preference/get_brand_feat.sql').read_text()).to_pandas()
		pcid_feat_df = ks.sql(Path('./brand_lterm_preference/get_pcid_feat.sql').read_text()).to_pandas()
		sterm_df = ks.sql(Path('./brand_lterm_preference/get_sterm.sql').read_text()).to_pandas()
		brd_df = ks.sql(Path('./brand_lterm_preference/get_brand.sql').read_text()).to_pandas()
		feat_idx_df = ks.sql(Path('./brand_lterm_preference/get_feat_idx.sql').read_text()).to_pandas()

		pcid_feat_dict = {}
		for row in pcid_feat_df.itertuples():
			pcid_feat_dict.setdefault(row.userid, {})['feat1'] = row.feat1
			pcid_feat_dict.setdefault(row.userid, {})['feat2'] = row.feat2

		brand_feat_dict = {}
		for row in brand_feat_df.itertuples():
			brand_feat_dict.setdefault(row.brd_id, {})['feat1'] = row.feat1
			brand_feat_dict.setdefault(row.brd_id, {})['feat2'] = row.feat2


		target_dict = {'userid': [], 'brand': []}
		file_name = "/".join([self.OPT_HOME, 'sterm.txt'])
		try:
			txt_file = open(file_name, 'w')
			for row in sterm_df.itertuples():
				vec = []
				if (str(row.userid) in pcid_feat_dict) & (str(row.brand) in brand_feat_dict):
					ismatch = len(
						set(pcid_feat_dict[str(row.userid)]['feat2']) & set(brand_feat_dict[str(row.brand)]['feat2']))
					pcid_feat = pcid_feat_dict[str(row.userid)]['feat1']
					brand_feat = brand_feat_dict[str(row.brand)]['feat1']
					vec.append(pcid_feat)
					vec.append(brand_feat)
					if ismatch > 0:
						vec.append(str(feat_idx_df.iloc[0]['ismatch']) + ":" + str(1))
					txt_file.write("%s\n" % " ".join(vec))
					target_dict['userid'].append(row.userid)
					target_dict['brand'].append(row.brand)

			txt_file.close()
		except IOError as e:
			print("I/O error({0}) : {1}]".format(e.errno, e.strerror))
		else:
			if self.run_model() == 0:
				out_file = "/".join([self.OPT_HOME, 'sterm.txt.out'])
				score_df = pd.read_csv(out_file, sep='\t', names=["score"])

				target_df = pd.DataFrame(target_dict)
				rerank_df = pd.concat([target_df, score_df], axis=1)

				rerank_df['brand'] = rerank_df['brand'].astype(int)
				brd_df['brand'] = brd_df['brand'].astype(int)

				rerank_df = rerank_df.merge(brd_df, on='brand', how='left')
				rerank_df = rerank_df[
					(rerank_df.brand.notnull()) & (rerank_df.brd_nm.notnull()) & (rerank_df.cnt.notnull())]

				rerank_df['RN'] = rerank_df.sort_values(['userid', 'score'], ascending=[True, False]) \
					                  .groupby(['userid']) \
					                  .cumcount() + 1
				rerank_df = rerank_df[rerank_df.RN < 9]
				rerank_df = rerank_df.sort_values(['userid', 'RN'], ascending=[True, True])
				rerank_df = rerank_df[['userid', 'brand', 'score', 'brd_nm', 'cnt']]
				rerank_df['brand'] = rerank_df['brand'].astype(str)
				rerank_df['score'] = rerank_df['score'].astype(str)
				rerank_df['brd_nm'] = rerank_df['brd_nm'].astype(str)
				rerank_df['cnt'] = rerank_df['cnt'].astype(str)
				final_df = rerank_df.groupby('userid').agg(
					lambda x: '^'.join(list(x)) if x.name == 'brd_nm' else ','.join(list(x))).reset_index()
				final_df['prdid'] = ''

				self.df_to_table(final_df, "members_bycho.brand_preference_reranking")
			else:
				print("fail to create reranking file!!")