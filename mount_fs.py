from src.echofs import ECHOFS

# Define your actual AGI-OS path
mount_path = "C:/Users/machine.god-alpha/Desktop/AGI-OS"

# Initialize and start ECHOFS
fs = ECHOFS(mount_path)

print("\n[✔] ECHOFS Virtual Layer is now mounted at:")
print(f"    {mount_path}\n")

# List contents
fs.list()
