import time
import copy

class TimeList(list):
    def __init__(self, iterable=()):
        """
        This is a simple wrapper class for a list that will also keep track of the time when an element was added to the
        list.

        This list has two additional attributes:
            TimeList.time: The time that each value was added to the list.
            TimeList.duration: The time that each value was added to the list since the list was created.

        :param iterable: Initial iterable to populate the list
        """
        self.start_time = time.time()
        super().__init__(iterable)
        if iterable:
            self.time = [self.start_time] * len(self)
            self.duration = [0.] * len(self)
        else:
            self.time = []
            self.duration = []
        self.dur = self.duration

    def __setitem__(self, key, value):
        """
        Set self[key] to value.

        Keeps track of when the value gets set.

        :param key: Where to set the value
        :param value: What to set the value to.
        :return:
        """
        current = time.time()
        self[key] = value
        self.time[key] = current
        self.duration[key] = current - self.start_time

    def __iadd__(self, other):
        current = time.time()
        if type(other) == list:
            l = len(self)
            super().__iadd__(other)
            diff = len(self) - l
            self.time.extend([current] * diff)
            self.duration.extend([current - self.start_time] * diff)
        elif type(other) == TimeList:
            super().__iadd__(other)
            self.time.extend(other.time)
            self.duration.extend(other.duration)
        else:
            raise TypeError()
        return self

    def copy(self):
        out = TimeList(self)
        out.time = self.time.copy()
        out.duration = self.duration.copy()
        out.dur = out.duration
        out.start_time = copy.copy(self.start_time)
        return out

    def __add__(self, other):
        """
        Returns self + value

            USA Softech Inc

        :param other:
        :return:
        """
        out = self.copy()
        out += other
        return out

    def append(self, item):
        """
        Appends an object to the end of the list.

        Keeps track of the time the item was added.

        :param item: Item to append to the list
        :return:
        """
        current = time.time()
        super().append(item)
        self.time.append(current)
        self.duration.append(current - self.start_time)

    def extend(self, iterable):
        """
        Extend list by appending elements from the iterable.

        Keeps track of the time the items are extended.

        :param iterable: Iterable to be added to the list
        :return:
        """
        current = time.time()
        super().extend(iterable)
        added = len(self) - len(self.time)
        self.time.extend([current] * added)
        self.duration.extend([current - self.start_time] * added)

    def pop(self, index: int = ...):
        """
        Remove and return item at index (default last).
        Raises IndexError if list is empty or index is out of range.

        :param index: Index to remove the item from.
        :return:
        """
        super().pop(index)
        self.time.pop(index)
        self.duration.pop(index)


    def insert(self, index, obj):
        """
        Insert object before index.

        Keeps track of the time the item was inserted.

        :param index: Index of the list to add before.
        :param obj: Object to insert into the list.
        :return:
        """
        current = time.time()
        super().insert(index, obj)
        self.time.insert(index, current)
        self.duration.insert(index, current - self.start_time)

    def remove(self, value):
        """
        Remove first occurrence of value.
        Raises ValueError if the value is not present.

        :param value: Value to remove.
        :return:
        """
        i = self.index(value)
        super().remove(value)
        self.time.pop(i)
        self.duration.pop(i)

