from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.handler.csv_handler import CSVHandler

def foo():
    for i in range(10000):
        print("It: " + str(i+1))

def bar():
    for i in range(10000):
        print("It2: " + str(i+1))

csv_handler = CSVHandler('result.csv')

with EnergyContext(handler=csv_handler, domains=[RaplPackageDomain(0)], start_tag='foo') as ctx:
    foo()
    ctx.record(tag='bar')
    bar()

csv_handler.save_data()