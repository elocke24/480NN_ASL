import random

class ASLLearning:
    def __init__(self):
        # make list of all chars (A-Z) minus j and z
        self.charList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y']

        # current list is our temp list to hold snuffled chars
        self.currentList = self.charList

        # keep track of current point in the list
        self.currentIndex = 0

        # list of correct and incorrect
        self.correct = []
        self.incorrect = []

    # shuffle list and put index back at zero

    def ShuffleList(self): # just shuffles
        random.shuffle(self.currentList)

    def ResetList(self): # shuffles and resets
        random.shuffle(self.currentList)
        self.currentIndex = 0

    # get current char in list
    def GetCurrentChar(self):
        if self.currentIndex < len(self.currentList):
            return self.currentList[self.currentIndex]
        return None # end of list

    def SetNextChar(self):
        self.currentIndex += 1

    # does not move to next
    def GetNextChar(self):
        if self.currentIndex + 1 < len(self.currentList):
            return self.currentList[self.currentIndex + 1]
        else:
            return None


    def CheckPred(self, predChar):
        target = self.GetCurrentChar()
        if target is None:
            return False

        # format to upper
        predCharFormatted = str(predChar).upper()

        matching = (predCharFormatted == target)

        # add letter we got right to list
        if matching:
            self.correct.append(target)
        else:
            # add letter we got wrong to list
            # we can pull last char in list right away to show what the letter was supposed to be
            self.incorrect.append(target)

        return matching

    # clear incorrect list
    def clear_incorrect_list(self):
        self.incorrect = []

    # clear correct list
    def clear_correct_list(self):
        self.correct = []

    # return progress of current list
    def get_progress(self):
        return f"{self.currentIndex + 1}/{len(self.currentList)}"


if __name__ == "__main__":
    # debug block
    session = ASLLearning()
    print(f"Target: {session.GetCurrentChar()}")

    # simulate correct guess
    print(f"Guessing {session.GetCurrentChar()}...")
    result = session.CheckPred(session.GetCurrentChar())
    print(f"Result: {result}")

    # move next
    print(f"Next Target: {session.GetNextChar()}")

    session.ShuffleList()
    print(f"Next Target: {session.GetNextChar()}")