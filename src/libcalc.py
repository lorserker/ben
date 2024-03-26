import os
import clr

BGADLL_PATH = os.path.join('../bin', 'BGADLL')

# Load the .NET assembly and import the types and classes from the assembly
clr.AddReference(BGADLL_PATH)
from BGADLL import DDS, Macros

d2 = DDS("KQ987.KJ3.AT6.8 62.AT72.K82.J74 .Q984.Q97543.KQ AJT543.65.J.T95", Macros.Trump.Diamond, Macros.Player.West)
d2.Execute("JD" + " x")
print(d2)
print(d2.Tricks("8D"))

d2 = DDS("KQ987.KJ3.AT6. 62.AT72.K82.J7 .Q984.Q97543.K AJT543.65.J.T9", Macros.Trump.Diamond, Macros.Player.West)
d2.Execute("TC" + " x")
print(d2)
print(d2.Tricks("8D"))
