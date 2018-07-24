import re

from py_stringmatching import utils
from py_stringmatching.tokenizer.definition_tokenizer import DefinitionTokenizer


class NumericTokenizer(DefinitionTokenizer):
    """Returns tokens that are maximal sequences of consecutive numeric characters. 

    Args:
        return_set (boolean): A flag to indicate whether to return a set of
                              tokens instead of a bag of tokens (defaults to False).
                              
    Attributes: 
        return_set (boolean): An attribute to store the value of the flag return_set.
    """
    
    def __init__(self, return_set=False):
        self.__num_regex = re.compile('[0-9]+')
        super(NumericTokenizer, self).__init__(return_set)

    def tokenize(self, input_string):
        """Tokenizes input string into numeric tokens.

        Args:
            input_string (str): The string to be tokenized.

        Returns:
            A Python list, which represents a set of tokens if the flag return_set is true, and a bag of tokens otherwise. 

        Raises:
            TypeError : If the input is not a string.
        """
        utils.tok_check_for_none(input_string)
        utils.tok_check_for_string_input(input_string)

        token_list = list(filter(None,
                                 self.__num_regex.findall(input_string)))

        if self.return_set:
            return utils.convert_bag_to_set(token_list)

        return token_list
