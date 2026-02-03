#!/usr/bin/env python3
"""Audio device detection and configuration script.

This script helps detect available audio input devices and configure
the appropriate device for the Speech-to-Braille system.

Usage:
    python scripts/detect_audio.py              # Interactive mode
    python scripts/detect_audio.py --auto       # Auto-select default
    python scripts/detect_audio.py --list       # Just list devices
"""

import sys
import argparse
from pathlib import Path

# Add src to path so we can import pyaudio
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio not installed. Run ./scripts/build.sh first.")
    sys.exit(1)


def list_audio_devices():
    """List all available audio devices with their capabilities."""
    p = pyaudio.PyAudio()

    print("\n" + "="*70)
    print("AVAILABLE AUDIO DEVICES")
    print("="*70 + "\n")

    input_devices = []
    default_input = None

    try:
        default_input = p.get_default_input_device_info()
    except:
        pass

    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            max_input = info.get('maxInputChannels', 0)
            max_output = info.get('maxOutputChannels', 0)

            # Determine device type
            if max_input > 0 and max_output > 0:
                device_type = "INPUT/OUTPUT"
                input_devices.append(i)
            elif max_input > 0:
                device_type = "INPUT ONLY"
                input_devices.append(i)
            elif max_output > 0:
                device_type = "OUTPUT ONLY"
            else:
                device_type = "UNKNOWN"

            # Check if this is the default input
            is_default = ""
            if default_input and info['index'] == default_input['index']:
                is_default = " [DEFAULT INPUT]"

            # Print device info
            print(f"Device {i}: {info['name']}{is_default}")
            print(f"  Type: {device_type}")
            print(f"  Max Input Channels: {max_input}")
            print(f"  Max Output Channels: {max_output}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']:.0f} Hz")
            print()

        except Exception as e:
            print(f"Device {i}: Error reading device info - {e}\n")

    p.terminate()

    return input_devices, default_input


def get_user_choice(input_devices, default_input):
    """Prompt user to select an input device."""
    if not input_devices:
        print("ERROR: No input devices found!")
        return None

    print("="*70)
    print("SELECT INPUT DEVICE")
    print("="*70 + "\n")

    if default_input:
        print(f"Default input device: {default_input['index']} - {default_input['name']}")
        print()

    print("Available input devices:")
    for idx in input_devices:
        print(f"  {idx}")
    print()

    while True:
        try:
            choice = input(f"Enter device number (or press Enter for default): ").strip()

            if choice == "" and default_input:
                return default_input['index']

            choice = int(choice)
            if choice in input_devices:
                return choice
            else:
                print(f"Invalid choice. Must be one of: {input_devices}")
        except ValueError:
            print("Invalid input. Enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return None


def save_device_config(device_index):
    """Save device configuration to .env file."""
    env_file = Path(__file__).parent.parent / ".env"

    # Read existing .env if it exists
    env_lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = [line for line in f.readlines()
                        if not line.startswith('AUDIO_DEVICE_INDEX=')]

    # Add new device index
    env_lines.append(f'AUDIO_DEVICE_INDEX={device_index}\n')

    # Write back
    with open(env_file, 'w') as f:
        f.writelines(env_lines)

    print(f"\n✓ Saved device configuration to {env_file}")
    print(f"  AUDIO_DEVICE_INDEX={device_index}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and configure audio input device for Speech-to-Braille"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all devices and exit'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-select default input device'
    )
    parser.add_argument(
        '--device',
        type=int,
        help='Directly set device index (no prompts)'
    )

    args = parser.parse_args()

    # List devices
    input_devices, default_input = list_audio_devices()

    if args.list:
        return 0

    # Determine device index
    if args.device is not None:
        device_index = args.device
        if device_index not in input_devices:
            print(f"ERROR: Device {device_index} is not a valid input device.")
            return 1
    elif args.auto:
        if default_input:
            device_index = default_input['index']
            print(f"Auto-selected default input: Device {device_index}")
        else:
            print("ERROR: No default input device found. Use interactive mode.")
            return 1
    else:
        # Interactive mode
        device_index = get_user_choice(input_devices, default_input)
        if device_index is None:
            return 1

    # Save configuration
    save_device_config(device_index)

    print("\n" + "="*70)
    print("CONFIGURATION COMPLETE")
    print("="*70)
    print("\nThe audio device has been configured.")
    print("Run './scripts/run.sh' to start the application.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
