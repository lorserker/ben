from ddsolver import ddsolver
import time

dd = ddsolver.DDSolver(dds_mode=1)
t_start = time.time()
        
# Create PBN for hand
hands_pbn = ['N:7.T87.K876543.97 T6.AK9654..T8654 KQJ5.QJ3.QT92.KJ A98432.2.AJ.AQ32']
dd_solved = dd.solve(1, 1, [], hands_pbn, 1)
print(dd_solved)
dd_solved = dd.solve(5, 1, [], hands_pbn, 2)
print(dd_solved)
dd_solved = dd.solve(3, 1, [], hands_pbn, 3)
print(dd_solved)

hands_pbn = ['N:A753.J72.964.Q6 KQ964.AK.AKT7.87 JT.65.Q853.K9542 82.QT9843.J2.AJT']
dd_solved = dd.solve(2, 0, [50], hands_pbn, 1)
print(dd_solved)
