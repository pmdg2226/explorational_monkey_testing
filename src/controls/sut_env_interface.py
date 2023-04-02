

from abc import ABC, abstractmethod


class SutEnvironment(ABC):
    '''System Under Test (SUT) Environment abstract class'''

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initial_state(self):
        '''Ensures controlled SUT is in the defined initial state.'''
        pass

    @abstractmethod
    def perform_action(self):
        '''Performs an action on the SUT'''
        pass

    @abstractmethod
    def deliver_feedback(self):
        '''Returns feedback information from SUT'''
        pass

