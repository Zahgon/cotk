'''
A module for word vector
'''
import numpy as np
import os
from .._utils.metaclass import DocStringInheritor, LoadClassInterface
from typing import List, Dict, Union, Optional, Any

from ..file_utils import get_resource_file_path


class WordVector(LoadClassInterface, metaclass=DocStringInheritor):
	r'''Base of all word vector loader.
	'''

class GeneralWordVector(WordVector):
	r'''Bases: :class:`.dataloader.WordVector`

	This class is a general pretrained word vector.

	Arguments:
		{FILE_ID_DOCS} {_FILE_ID_DEFAULT}

	{INPUT_FORMAT}
	'''

	FILE_ID_DOCS = r'''
		file_id (str, ``None``): A str indicates the source of word vectors. It can be local path (``"./data"``), a resource name
				(``"resources://dataset"``), or an url (``"http://test.com/dataset.zip"``).
				See :meth:`cotk.file_utils.get_resource_file_path` for further details.
				If ``None``, do not use pretrained word vector.'''
	_FILE_ID_DEFAULT = ""

	INPUT_FORMAT = r'''
	Input Format
		A text file named ``wordvec.txt`` should be contained in the path. In the file, each word vec should be
		described in two lines. The first line is the word (or phrase), then the next line is multiple floats
		indicating the embedding.

		Example of ``wordvec.txt``:

		.. code-block:: none

			word
			0.0 1.0 -2.3
			phrases
			0.3 -1.2 3.4
	'''

	def __init__(self, file_id: Union[str, None]):
		super().__init__()
		self.file_id: Optional[str] = file_id
		self.file_path = get_resource_file_path(file_id) if file_id else None

	def _load_raw_word2vec(self) -> Dict[str, str]:
		'''Load raw word vectors from file.
		'''
		pass

	def load_matrix(self, n_dims: int, vocab_list: List[str], \
			mean: Optional[Union[float, List, np.ndarray]] = None, \
			std: Optional[Union[float, List, np.ndarray]] = None, \
			default_embeddings: Optional[Union[List, np.ndarray]] = None) -> np.ndarray:
		r'''Load pretrained word vector and return a numpy 2-d array. The ith row is the feature
		of the ith word in ``vocab_list``. If some feature is not included in pretrained
		word vector, it will be initialized by:

		* ``default_embeddings``, if it is not ``None``.
		* normal distribution with ``mean`` and ``std``, otherwise.

		Arguments:
			n_dims (int): specify the dimension size of word vector. If ``n_dims``
				is bigger than size of pretrained word vector, the rest embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the embedding will be
				initialized by ``default_embeddings`` or a normal distribution.
			mean (float, Any, None): The mean of normal distribution.
				It can be a float, or an array whose shape is ``[n_dims]``.
				if ``None``, it will be set by the mean of loaded word vector embedding.
				Default: ``None``.
			std (float, Any, None): The standard deviation of normal distribution.
				It can be a float, or an array whose shape is ``[n_dims]``.
				if ``None``, it will be set by the standard deviation of loaded word vector embedding.
				Default: ``None``.
			default_embeddings (Any, optional): The default embeddings, its size should be
				``[len(vocab_list), n_dims]``. Default: None, which indicates initializing
				the embeddings from the normal distribution with ``mean`` and ``std``.

		Returns:

			(:class:`numpy.ndarray`): A  2-d array. Size:``[len(vocab_list), n_dims]``.
		'''
		pass

	def load_dict(self, vocab_list: List[str]) -> Dict[str, np.ndarray]:
		r'''Load word vector and return a dict that maps words to vectors.

		Arguments:
			vocab_list (list): specify the vocab list used in data loader. If there
				is any word not appeared in pretrained word vector, the feature will
				not be returned.

		Returns:

			(dict): maps a word (str) to its pretrained embedding (:class:`numpy.ndarray`)
				where its shape is [ndims].
		'''
		pass


class Glove(GeneralWordVector):
	r'''Bases: :class:`.dataloader.GeneralWordVector`, :class:`.dataloader.WordVector`

	GloVe is pre-trained word vector named `Global Vectors for Word Representation`.

	References:

		[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
		GloVe: Global Vectors for Word Representation.

	Arguments:
		{FILE_ID_DOCS} {_FILE_ID_DEFAULT}

	'''
	_FILE_ID_DEFAULT = "Default: ``resources://Glove300d``.	A 300-d pretrained GloVe will be downloaded (or loaded from cache) and used."

	def __init__(self, file_id="resources://Glove300d"):
		super().__init__(file_id=file_id)
