'''A module for vocab'''
from typing import Optional, List, Dict, Any
from collections import Counter, OrderedDict
from itertools import chain
import logging
import hashlib

from .._utils.typehint import OrderedDictType
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from .._utils.unordered_hash import dumps
from .context import VocabContext
from .tokenizer import PretrainedTokenizer


class Vocab(LoadClassInterface, metaclass=DocStringInheritor):
	'''A class for storing vocabulary.
	This is an abstract base class.
	It often works as a part of :class:`Field` or is shared between :class:`Field`.

	See :ref:`introduction of vocabulary<vocabulary_ref>` for more information.

	Arguments:

		This class do not contains arguments for initialization.
	'''

	NOT_SPECIFIED_DOCS = r'''
	If any argument is not specified,
	the value will be first retrieved from :class:`VocabContext`. If still ``None``, default
	value will be used.
	'''

	def __init__(self):
		if self.__class__.__name__ == "Vocab":
			raise NotImplementedError("This class is an abstract class, use GeneralVocab instead.")
		self._setting_hash: Optional[str] = None

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		'''Add tokens for this vocabulary instance, the tokens will be used for building
		vocabulary list. Must be called before :meth:`.build_vocab`.

		Arguments:
			tokens (List[str]): A list of tokens to add to the vocabulary.
			vocab_from (str): One of ``train``, ``test``, ``extra``.

				* ``train``: The tokens are from the training data. Frequent vocabs are selected from tokens of this type.
				* ``test``: The tokens are from the validation data or test data. Rare vocabs are selected from tokens of this type.
				* ``extra``: The tokens are from extra data. The tokens of this type will not selected as frequent or rare vocabs.
		'''
		raise NotImplementedError

	def build_vocab(self):
		'''Building the vocabulary list according to the tokens from :meth:`.add_tokens`.
		'''
		raise NotImplementedError

	_VOCAB_MORE_DOCSTRING = ""
	CONVERT_TOKENS_TO_IDS_ARG = """
			tokens (List[str]): List of tokens.
			only_frequent_word (bool, optional): Use ``unk`` for rare tokens. Defaults: False.
	"""
	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		'''Convert list of tokens to list of ids. {_VOCAB_MORE_DOCSTRING}

		Arguments:{CONVERT_TOKENS_TO_IDS_ARG}
		'''
		raise NotImplementedError

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		'''Convert list of ids to list of tokens. {_VOCAB_MORE_DOCSTRING}

		Arguments:
			ids (List[int]): List of ids.
		'''
		raise NotImplementedError

	@property
	def frequent_vocab_size(self):
		'''int: The number of **frequent** words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def all_vocab_size(self):
		'''int: The number of frequent words and rare words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def frequent_vocab_list(self):
		'''list: The list of frequent words. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	@property
	def all_vocab_list(self):
		'''list: The list of frequent words and rare words. Frequent words are always in the front of the list. {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	SPECIAL_TOKEN_DOCS = '''Special tokens mapping is an ordered dict \
		mapping the general name of special tokens to its string. \
		The key must be one of the following: \
		``pad``, ``unk``, ``go``, ``eos``, ``sep``, ``cls``, ``mask``. \
		The value can be arbitrary string, e.g., ``"<pad>"``, ``"<unk>"``.'''

	def get_special_tokens_mapping(self) -> OrderedDictType[str, str]:
		'''Get special tokens mapping. {SPECIAL_TOKEN_DOCS} {_VOCAB_MORE_DOCSTRING}
		'''
		raise NotImplementedError

	def get_special_tokens_id(self, name: str) -> int:
		'''Get id of special token specifying the general name.
		Raise ``KeyError`` if no such token in this instance. {_VOCAB_MORE_DOCSTRING}

		Arguments:
			name (str): the general name, must be one of the following,
				``pad``, ``unk``, ``go``, ``eos``, ``sep``, ``cls``, ``mask``.
		'''
		raise NotImplementedError

	@property
	def pad_id(self) -> int:
		'''int: The id of pad token. Raise ``KeyError`` if no pad token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		pass
	@property
	def unk_id(self) -> int:
		'''int: The id of unk token. Raise ``KeyError`` if no unk token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		pass
	@property
	def go_id(self) -> int:
		'''int: The id of go token. Raise ``KeyError`` if no go token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		pass
	@property
	def eos_id(self) -> int:
		'''int: The id of eos token. Raise ``KeyError`` if no eos token in this instance. {_VOCAB_MORE_DOCSTRING}
		'''
		pass

	def get_setting_hash(self) -> str:
		'''Get setting hash for the Vocabulary instance.
		See :ref:`here <dataloader_hash_ref>` for the explaination of ``setting hash``.
		'''
		assert self._setting_hash is not None
		return self._setting_hash

	def get_vocab_hash(self) -> str:
		'''Get vocab hash for the Vocabulary instance.
		See :ref:`here <dataloader_hash_ref>` for the explaination of ``vocab hash``.
		'''
		raise NotImplementedError

class GeneralVocab(Vocab):
	'''Bases: :class:`.dataloader.Vocab`

	A vocabulary class for general use.

	This class always have the following 4 speical tokens: ``pad``, ``unk``, ``go``, ``eos``.

	{NOT_SPECIFIED_DOCS}

	Arguments:
			{MIN_FREQUENT_VOCAB_TIMES_DOCS} {MIN_FREQUENT_VOCAB_TIMES_DEFAULT}
			{MIN_RARE_VOCAB_TIMES_DOCS} {MIN_RARE_VOCAB_TIMES_DEFAULT}
			{SPECIAL_TOKEN_DOCS} {SPECIAL_TOKEN_DEFAULT}
			special_appeared_in_data (bool, optional): If the string of special tokens will
					appear in the data. Default: If not specified, it will be ``False``.
	'''

	MIN_FREQUENT_VOCAB_TIMES_DOCS = r"""
			min_frequent_vocab_times (int, optional): Tokens from training data appeared
					no less than ``min_frequent_vocab_times`` will be regarded as frequent words."""
	MIN_FREQUENT_VOCAB_TIMES_DEFAULT = r"""Default: ``0``"""
	MIN_RARE_VOCAB_TIMES_DOCS = r"""
			min_rare_vocab_times (int, optional): Tokens from training data or test data
					appeared more than ``min_rare_vocab_times`` will be regarded as rare words
					(frequent word excluded). """
	MIN_RARE_VOCAB_TIMES_DEFAULT = r"""Default: ``0``"""

	SPECIAL_TOKEN_DOCS = r"""
			special_tokens_mapping (OrderedDict, optional): {Vocab.SPECIAL_TOKEN_DOCS}
					It must at least contains ``pad``, ``unk``, ``go``, ``eos``.
					All the value of special tokens cannot be the same."""
	SPECIAL_TOKEN_DEFAULT = r"""Default: If ``None``, it will be ``OrderedDict([("pad", "<pad>"), ("unk", "<unk>"), ("go", "<go>"), ("eos", "<eos>")]``."""

	def __init__(self, min_frequent_vocab_times: Optional[int] = None, \
			min_rare_vocab_times: Optional[int] = None, \
			special_tokens_mapping: Optional[OrderedDictType[str, str]] = None, \
			special_appeared_in_data: Optional[bool] = None):
		super().__init__()

		with VocabContext.set_parameters(\
				min_frequent_vocab_times=min_frequent_vocab_times,\
				min_rare_vocab_times=min_rare_vocab_times,\
				special_tokens_mapping=special_tokens_mapping,\
				special_appeared_in_data=special_appeared_in_data):
			self.min_frequent_vocab_times: int = VocabContext.get("min_frequent_vocab_times", 0)
			self.min_rare_vocab_times: int = VocabContext.get("min_rare_vocab_times", 0)
			filled_special_tokens: Optional[OrderedDictType[str, str]] = VocabContext.get("special_tokens_mapping", None)
			self.special_appeared_in_data: bool = VocabContext.get("special_appeared_in_data", False)

		self.special_tokens_mapping = filled_special_tokens or OrderedDict(
			[("pad", "<pad>"), ("unk", "<unk>"), ("go", "<go>"), ("eos", "<eos>")]
		)

		if {"pad", "unk", "go", "eos"}.difference(set(self.special_tokens_mapping.keys())):
			raise ValueError("Special tokens should at least contains 4 tokens: pad, unk, go, eos.")
		if set(self.special_tokens_mapping.keys()).difference({"pad", "unk", "go", "eos", "sep", "cls", "mask"}):
			raise ValueError("Special tokens should not contains keys other than pad, unk, go, eos, sep, cls, mask.")
		if len(set(self.special_tokens_mapping.values())) != len(set(self.special_tokens_mapping.keys())):
			raise ValueError("All the value of special tokens cannot be the same.")

		self.mode = "init"
		self.train_tokens: Optional[List[str]] = []
		self.test_tokens: Optional[List[str]] = []

		self._all_vocab_list: Optional[List[str]] = None
		self.word2id: Optional[Dict[str, int]] = None
		self._frequent_vocab_size: int = 0

		self._setting_hash = hashlib.sha256(dumps([ \
			"Vocab", \
			"configs", \
			self.min_frequent_vocab_times, \
			self.min_rare_vocab_times, \
			self.special_tokens_mapping, \
			self.special_appeared_in_data \
		])).hexdigest()

	@staticmethod
	def from_predefined(vocab_list: List[str], \
		frequent_vocab_size: int, \
		special_tokens_mapping: Optional[OrderedDictType[str, str]] = None) -> "GeneralVocab":
		'''Return a :class:`GeneralVocab` instance, whose vocabulary comes from a predefined list. See :meth:`.from_predefined_vocab` if
		you want to use the vocabulary from an existing :class:`GeneralVocab` instance.

		Arguments:
			vocab_list (List[str]): A list of all vocabulary.
			frequent_vocab_size (int): the length of the frequent words.
					The frequent word must be in the front of the ``vocab_list``.
			{SPECIAL_TOKEN_DOCS} Special tokens MUST be in the front of the ``frequent_vocab_list`` (ordered sensitive).
					{SPECIAL_TOKEN_DEFAULT}
		'''
		pass

	@staticmethod
	def from_predefined_vocab(vocab: "GeneralVocab") -> "GeneralVocab":
		'''Return a new :class:`GeneralVocab` instance from ``vocab``. The new instance have the same
		vocabulary list as the old one.

		Arguments:
			vocab(:class:`GeneralVocab`): The old instance.
		'''
		pass


	@staticmethod
	def from_frequent_word(frequent_vocab_list: List[str], \
			special_tokens_mapping: Optional[OrderedDictType[str, str]] = None) -> "GeneralVocab":
		'''Return a :class:`GeneralVocab` instance, whose vocabulary comes from a predefined frequent list.
		And its rare word list can be built later.
		See :meth:`.from_frequent_word_of_vocab`
		if you want to use the frequent vocabulary from an existing :class:`GeneralVocab` instance.

		Arguments:
			frequent_vocab_list (List[str]): A list of frequent vocabulary.
			{SPECIAL_TOKEN_DOCS} Special tokens MUST be in the front of the ``frequent_vocab_list`` (ordered sensitive).
					{SPECIAL_TOKEN_DEFAULT}
		'''
		pass

	@staticmethod
	def from_frequent_word_of_vocab(vocab: "GeneralVocab") -> "GeneralVocab":
		'''Return a :class:`GeneralVocab` instance, which has the same frequent vocabulary list as the old one.
		The rare word list can be built later.

		Arguments:
			vocab(:class:`GeneralVocab`): The old instance to provide frequent words.

		'''
		pass

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		pass

	def build_vocab(self) -> None:
		pass

	def get_special_tokens_id(self, name) -> int:
		try:
			return self.word2id[self.special_tokens_mapping[name]] # type: ignore
		except KeyError:
			raise KeyError("No such special token in this class")

	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		if self.word2id is None:
			raise RuntimeError("You have to run build_vocab first")
		ids = [self.word2id.get(token, self.unk_id) for token in tokens]
		if only_frequent_word:
			ids = [self.unk_id if i >= self._frequent_vocab_size else i for i in ids]
		return ids

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		if self._all_vocab_list is None:
			raise RuntimeError("You have to run build_vocab first")
		return [self._all_vocab_list[word] for word in ids]

	def get_vocab_hash(self) -> str:
		return hashlib.sha256(dumps([ \
				self._all_vocab_list, \
				self._frequent_vocab_size, \
				len(self.special_tokens_mapping) \
			])).hexdigest()

	@property
	def frequent_vocab_size(self):
		pass

	@property
	def all_vocab_size(self):
		pass

	@property
	def frequent_vocab_list(self):
		pass

	@property
	def all_vocab_list(self):
		pass

	def get_special_tokens_mapping(self):
		return self.special_tokens_mapping

class PretrainedVocab(Vocab):
	'''Bases: :class:`.dataloader.Vocab`

	Use the vocabulary from a pretrained tokenizer in ``transformers`` package.
	This class is usually used for pretrained models, and it **do NOT** have rare words.

	Unlike :class:`GeneralVocab`, this class do not always have ``pad``, ``unk``, ``go``, ``eos``.
	Some special tokens may refer to the same token.

	Arguments:
		tokenizer (``transformers.PretrainedTokenizer``): A pretrained tokenizer from transformers package.
	'''
	def __init__(self, tokenizer: Any):
		super().__init__()
		self.tokenizer = PretrainedTokenizer(tokenizer)
		self._inner_tokenizer = tokenizer
		self._setting_hash = hashlib.sha256(dumps(["pretrained", self.tokenizer.get_setting_hash()])).hexdigest()

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		pass

	def build_vocab(self) -> None:
		pass

	def get_vocab_hash(self):
		return self._setting_hash #vocab hash is represented by tokenizer

	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		return self._inner_tokenizer.convert_tokens_to_ids(tokens)

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		return self._inner_tokenizer.convert_ids_to_tokens(ids)

	@property
	def frequent_vocab_size(self):
		pass

	@property
	def all_vocab_size(self):
		pass

	@property
	def frequent_vocab_list(self):
		pass

	@property
	def all_vocab_list(self):
		pass

	def get_special_tokens_mapping(self):
		old_key = ["pad_token", "unk_token", "bos_token", "eos_token", "sep_token", "cls_token", "mask_token"]
		new_key = ["pad", "unk", "go", "eos", "sep", "cls", "mask"]
		res = OrderedDict()
		for key, value in self._inner_tokenizer.special_tokens_map.items():
			if key in old_key:
				idx = old_key.index(key)
				res[new_key[idx]] = value
		return res

	def get_special_tokens_id(self, name):
		try:
			return self.convert_tokens_to_ids([self.get_special_tokens_mapping()[name]])[0]
		except KeyError:
			raise KeyError("No such special token in this class")

class SimpleVocab(Vocab):
	"""Bases: :class:`.dataloader.Vocab`

	A very simple vocabulary class. No rare vocabs or special tokens.
	Used by :class:`SparseLabel`.

	Arguments:
			This class do not contains arguments for initialization.

	"""
	def __init__(self):
		super().__init__()
		self._setting_hash = hashlib.sha256(
			dumps([self.__class__.__name__, "configs"])
		).hexdigest()
		self._token_counter = Counter()
		self._all_vocab_list: List[str] = None
		self.word2id: Dict[str, int] = None
		self.mode = "init"

	def add_tokens(self, tokens: List[str], vocab_from: str) -> None:
		pass

	add_tokens.__doc__ = Vocab.add_tokens.__doc__ + r"""
	Notes:
		Since frequency is not important in this class, argument `vocab_from` has no effect.
	"""

	def build_vocab(self):
		pass

	def convert_tokens_to_ids(self, tokens: List[str], only_frequent_word=False) -> List[int]:
		if self.word2id is None:
			raise RuntimeError("You have to run build_vocab first")
		return [self.word2id[token] for token in tokens]

	def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
		if self._all_vocab_list is None:
			raise RuntimeError("You have to run build_vocab first")
		return [self._all_vocab_list[i] for i in ids]

	@property
	def frequent_vocab_size(self):
		pass

	@property
	def all_vocab_size(self):
		pass

	@property
	def frequent_vocab_list(self):
		pass

	@property
	def all_vocab_list(self):
		pass

	def get_special_tokens_mapping(self) -> OrderedDictType[str, str]:
		return {}

	def get_special_tokens_id(self, name: str) -> int:
		raise NotImplementedError("SimpleVocab don\'t use any special tokens.")

	def get_vocab_hash(self) -> str:
		return hashlib.sha256(
			dumps([self._all_vocab_list])
		).hexdigest()
