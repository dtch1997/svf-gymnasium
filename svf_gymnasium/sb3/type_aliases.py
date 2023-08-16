import sys
import numpy as np

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    SupportsFloat,
    Tuple,
    Union,
)

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class SafetyFilter(Protocol):
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        nominal_action: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the safe action from an observation and nominal action (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
