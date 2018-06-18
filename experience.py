import random
from collections import namedtuple

Memento = namedtuple("Memento", "frames instruction action value_return action_probs")

class ExperienceReplay:
    def __init__(self, capaciy):
        self.mementos = []
        self.capacity = capaciy

    def _remove_leftovers(self, container, capacity):
        leftovers = len(container) - capacity
        if(leftovers > 0):
            del container[:leftovers]

        return container

    def _sample(self, container, n):
        if(n > len(container)):
            return random.sample(container, len(container))
        else:
            return random.sample(container, n)

    def append(self, memento: Memento):
        self.mementos.append(memento)
        self._remove_leftovers(self.mementos, self.capacity)

    def sample(self, n):
        return self._sample(self.mementos, n)
