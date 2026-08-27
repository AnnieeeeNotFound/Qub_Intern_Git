"""Step 0: Can Python see the AD3? """
import os, sys

DWFLIB_DIR = r"D:\Digilent\WaveForms3"   # where dwf.dll lives
os.environ["PATH"] = DWFLIB_DIR + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(DWFLIB_DIR)

from pydwf import DwfLibrary

dwf = DwfLibrary()
print("DWF library version:", dwf.getVersion())

n = dwf.deviceEnum.enumerateDevices()
print(f"Devices found: {n}")
for i in range(n):
    print(f"  [{i}] name={dwf.deviceEnum.deviceName(i)!r} "
          f"serial={dwf.deviceEnum.serialNumber(i)} "
          f"type={dwf.deviceEnum.deviceType(i)} "
          f"opened={dwf.deviceEnum.deviceIsOpened(i)}")

if n == 0:
    print("\nNo Digilent device visible. Check USB cable / close WaveForms GUI.")
    sys.exit(1)

print("\nPYDWF LINK OK - Python can talk to your AD3.")
