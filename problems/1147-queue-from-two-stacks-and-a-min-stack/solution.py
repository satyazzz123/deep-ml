class Queue:
    def __init__(self):
        self.items = []
        self.front = 0

    def is_empty(self):
        return self.front >= len(self.items)

    def enqueue(self, element):
        self.items.append(element)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")

        value = self.items[self.front]
        self.front += 1
        return value


class SpecialStack:
    def __init__(self):
        self.items = []

    def mpush(self, value):
        self.items.append(value)

    def mpop(self):
        return self.items.pop()

    def mtop(self):
        return self.items[-1]

    def mmin(self):
        return min(self.items)

def process_operations(operations):
    queue = Queue()
    stack = SpecialStack()

    outputs = []

    for operation in operations:
        op = operation[0]

        if op == "enqueue":
            queue.enqueue(operation[1])

        elif op == "dequeue":
            outputs.append(queue.dequeue())

        elif op == "mpush":
            stack.mpush(operation[1])

        elif op == "mpop":
            outputs.append(stack.mpop())

        elif op == "mtop":
            outputs.append(stack.mtop())

        elif op == "mmin":
            outputs.append(stack.mmin())

    return outputs