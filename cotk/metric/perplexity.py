r"""
Containing some classes and functions about perplexity evaluating results of models.
"""
import random
import numpy as np
from typing import Union, List, Any, Optional, Dict

from .._utils.imports import LazyObject, LazyModule
from ..dataloader import LanguageProcessing, Sentence
from .metric import MetricBase

torch = LazyModule("torch", globals())
torch.Tensor = LazyObject("torch.Tensor") #type: ignore

class PerplexityMetric(MetricBase):
	'''Metric for calculating perplexity.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.REFERENCE_LEN_KEY_ARGUMENTS}
		{MetricBase.GEN_LOG_PROB_KEY_ARGUMENTS}
		{MetricBase.GENERATE_RARE_VOCAB_ARGUMENTS}
		{MetricBase.FULL_CHECK_ARGUMENTS}

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> reference_allvocabs_key="ref_allvocabs"
		>>> reference_len_key="ref_length"
		>>> gen_log_prob_key="gen_log_prob"
		>>> metric = cotk.metric.PerplexityMetric(dl,
		...	    reference_allvocabs_key=reference_allvocabs_key,
		...	    reference_len_key=reference_len_key,
		...	    gen_log_prob_key=gen_log_prob_key)
		>>> data = {
		...	    reference_allvocabs_key: [[2, 10, 64, 851, 3], [2, 10, 48, 851, 3]],
		...	    # reference_allvocabs_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
		...     reference_len_key: [5, 5],
		...     gen_log_prob_key: [[[-11.31, -11.31,  -0.69, ..., -11.31, -11.31, -11.31],...],...] # shape == (batch, length, vocab_size)
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'perplexity': 81458.00000000006,
 		 'perplexity hashvalue': '7f9b88b8f9996f5d49a512258f250fbc56adee714952b2c696c0b36cce36f648'}
	'''

	_name = 'PerplexityMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
					   reference_allvocabs_key: str = "ref_allvocabs", \
					   reference_len_key: str = "ref_length", \
					   gen_log_prob_key: str = "gen_log_prob", \
					   generate_rare_vocab: bool = False, \
					   full_check: bool = False \
			  ):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.word_loss = 0
		self.length_sum = 0
		self.generate_rare_vocab = generate_rare_vocab
		self.full_check = full_check
		self.engine_version = "unknown" # after first forward, it will be filled with 'default' or 'pytorch'

		self.resp: List[str] = []
		#self.resp_length = []
		self.gen_valid_log_prob: List[np.ndarray] = []
		self.gen_unk_log_prob: List[np.ndarray] = []

		self.have_unk = "unk" in self.dataloader.get_special_tokens_mapping()

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data. Smoothing will be performed for :ref:`rare vocabs <vocabulary_ref>`.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH}
				{MetricBase.FORWARD_REFERENCE_LEN_ARGUMENTS}
				* **data[gen_log_prob_key]** (list, :class:`numpy.ndarray`, :class:`torch.Tensor`):
				  The **log softmax** probability of the sentence generations model outputs.
				  A 3-d jagged or padded array of float.
				  Contains end token (eg:``<eos>``), but without start token (eg: ``<go>``).
				  Size: ``[batch_size, ~gen_sentence_length, vocab_size]`` for ``generate_rare_vocab = False``, or
				  ``[batch_size, ~gen_sentence_length, all_vocab_size]`` for ``generate_rare_vocab = True``,
				  where "~" means different sizes in this dimension is allowed.
				  If :class:`torch.Tensor` is used, the following data should also be
				  :class:`torch.Tensor`.

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]],
					...	    reference_len_key: [3,4],
					...	    gen_log_prob_key: [[[-3.80666249, -3.11351531, -2.7080502 , -2.42036813, -2.19722458,
							    -2.01490302, -1.86075234, -1.72722095, -1.60943791],...],...]
					... }
		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		pass

	def _normal_forward(self, resp_allvocabs, resp_length, gen_log_prob):
		pass

	def _pytorch_forward(self, resp_allvocabs, resp_length, gen_log_prob):
		pass

	@classmethod
	def _run_f(cls, ele):
		'''Auxiliary function for computing perplexity:

		Returns:

			* tuple: sum of log perplexity and sum of sentence length.
		'''
		pass

	def close(self) -> Dict[str, Any]:
		r'''Return a dict which contains

			* **perplexity**: perplexity value.
			* **perplexity hashvalue**: hash value for perplexity metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()

		if self.engine_version == "pytorch":
			# pytorch is finished when forward
			if self.length_sum == 0:
				raise RuntimeError("The metric has not been forwarded data correctly.")
		else:
			if not self.gen_valid_log_prob:
				raise RuntimeError("The metric has not been forwarded data correctly.")

			loader = self.dataloader
			unk_id = loader.unk_id if self.have_unk else None
			tasks = ((self.gen_valid_log_prob[i], self.gen_unk_log_prob[i], self.resp[i], \
							self.generate_rare_vocab, loader.frequent_vocab_size, loader.all_vocab_size, unk_id) \
							for i, _ in enumerate(self.gen_valid_log_prob))

			# Multiprocessing seems can't boost the speed
			# if len(self.gen_valid_log_prob) > 100:
			# 	pool = Pool(multiprocessing.cpu_count())
			# 	for ans in tqdm.tqdm(pool.imap_unordered(self.run_f, tasks, chunksize=20), \
			# 		total=len(self.gen_valid_log_prob)):
			# 		self.word_loss += ans[0]
			# 		self.length_sum += ans[1]
			# 	pool.close()
			# 	pool.join()
			# else:
			for ans in map(self._run_f, tasks):
				self.word_loss += ans[0]
				self.length_sum += ans[1]

			self.resp = []
			self.gen_valid_log_prob = []
			self.gen_unk_log_prob = []

		res.update({"perplexity": np.exp(self.word_loss / self.length_sum), \
				"perplexity hashvalue": self._hashvalue()})
		return res

class MultiTurnPerplexityMetric(MetricBase):
	'''Metric for calculating multi-turn perplexity.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_LEN_KEY_ARGUMENTS}
		{MetricBase.GEN_LOG_PROB_KEY_ARGUMENTS}
		{MetricBase.GENERATE_RARE_VOCAB_ARGUMENTS}
		{MetricBase.FULL_CHECK_ARGUMENTS}

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> multi_turn_reference_allvocabs_key = "multi_turn_ref_allvocabs"
		>>> multi_turn_reference_len_key = "multi_turn_ref_length"
		>>> multi_turn_gen_log_prob_key = "multi_turn_gen_log_prob"
		>>> metric = cotk.metric.MultiTurnPerplexityMetric(dl,
		...	    multi_turn_reference_allvocabs_key="multi_turn_ref_allvocabs",
		...	    multi_turn_reference_len_key="multi_turn_ref_length",
		...	    multi_turn_gen_log_prob_key="multi_turn_gen_log_prob")
		>>> data = {
		...	    multi_turn_reference_allvocabs_key: [[[2, 10, 64, 851, 3], [2, 10, 64, 479, 3]], [[2, 10, 64, 279, 1460, 3]]],
		...     # multi_turn_reference_allvocabs_key = [[["<go>", "I", "like", "python", "<eos>"],
		...     # 	["<go>", "I", "like", "java", "<eos>"]],
		...     # 	[["<go>", "I", "like", "machine", "learning", "<eos>"]]]
		...
		...	    multi_turn_reference_len_key: [[5, 5], [6]],
		...	    multi_turn_gen_log_prob_key: [[[[-11.30784283, -11.30784283,  -0.69312263, ..., -11.30784283, -11.30784283, -11.30784283], ...], ...], ...]
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'perplexity': 81458.00000000006,
 		 'perplexity hashvalue': '3a7647507f2e0d05a235c1d3a29515dc8885650884d625a5b76d305541dca685'}
	'''

	_name = 'MultiTurnPerplexityMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
					   multi_turn_reference_allvocabs_key: str = "multi_turn_ref_allvocabs", \
					   multi_turn_reference_len_key: str = "multi_turn_ref_length", \
					   multi_turn_gen_log_prob_key: str = "multi_turn_gen_log_prob", \
					   generate_rare_vocab: bool = False, \
					   full_check: bool = False \
			  ):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.multi_turn_reference_len_key = multi_turn_reference_len_key
		self.multi_turn_gen_log_prob_key = multi_turn_gen_log_prob_key
		self.generate_rare_vocab = generate_rare_vocab
		self.sub_metric = PerplexityMetric(dataloader, \
				reference_allvocabs_key="ref_allvocabs", \
				reference_len_key="ref_length", \
				gen_log_prob_key="gen_log_prob", \
				generate_rare_vocab=generate_rare_vocab, \
				full_check=full_check)

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS_WITH_TORCH}
				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_LEN_ARGUMENTS}
				* **data[multi_turn_gen_log_prob_key]** (list, :class:`numpy.ndarray`, \
					:class:`torch.Tensor`):
				  The **log softmax** probability of the sentence generations model outputs.
				  A 4-d jagged or padded array. **log softmax** probability.
				  Contains end token (eg:``<eos>``), but without start token (eg: ``<go>``).
				  Size: ``[batch_size, ~gen_sentence_length, vocab_size]`` for ``generate_rare_vocab = False``, or
				  ``[batch_size, ~gen_sentence_length, all_vocab_size]` for ``generate_rare_vocab = True``,
				  where "~" means different sizes in this dimension is allowed.
				  If :class:`torch.Tensor` is used, the following data should also be
				  :class:`torch.Tensor`.

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...     multi_turn_reference_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]],
					...	    multi_turn_reference_len_key: [[3, 4], [5]],
					...	    multi_turn_gen_log_prob_key: [[[[-3.80666249, -3.11351531, -2.7080502,
						        -2.42036813, -2.19722458, -2.01490302, -1.86075234, -1.72722095,
						        -1.60943791], ...], ...], ...]
					... }
		Warning:
			``data[multi_turn_gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(multi_turn_gen_log_prob_key), -1)`` equals
			``np.ones((batch_size, ~gen_sentence_length))``
		'''
		pass

	def close(self) -> Dict[str, Any]:
		r'''Return a dict which contains

			* **perplexity**: perplexity value.
			* **perplexity hashvalue**: hash value for perplexity metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()
		res.update(self.sub_metric.close())
		return res
