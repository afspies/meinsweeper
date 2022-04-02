
# class Configuration():
#     def __init__(self) -> None:
#         self.test = True
#         self.fish = 7
#         self.frog = "yes_please"

#     def __repr__(self):
#         cfg_attr = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
#         return " ".join([f"{attr}={getattr(self,str(attr))}" for attr in cfg_attr])

# c = Configuration()
# print(c)