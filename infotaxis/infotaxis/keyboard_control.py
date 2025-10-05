#!/usr/bin/env python3

import rclpy
import sys
import termios
import tty
from std_msgs.msg import String


def get_key():
    """Get a single keypress from stdin"""
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('keyboard_control')
    pub = node.create_publisher(String, '/infotaxis/keyboard_command', 10)

    print('\n=== Keyboard Control for Infotaxis ===')
    print('Use arrow keys or WASD to move:')
    print('  UP/W    - Move up (+y)')
    print('  DOWN/S  - Move down (-y)')
    print('  LEFT/A  - Move left (-x)')
    print('  RIGHT/D - Move right (+x)')
    print('  P       - Print current position')
    print('  Q       - Quit')
    print('=====================================\n')

    try:
        while rclpy.ok():
            key = get_key()

            msg = String()
            if key in ['w', 'W']:
                msg.data = 'up'
                pub.publish(msg)
                print('→ Moving UP')
            elif key in ['s', 'S']:
                msg.data = 'down'
                pub.publish(msg)
                print('→ Moving DOWN')
            elif key in ['a', 'A']:
                msg.data = 'left'
                pub.publish(msg)
                print('→ Moving LEFT')
            elif key in ['d', 'D']:
                msg.data = 'right'
                pub.publish(msg)
                print('→ Moving RIGHT')
            elif key in ['p', 'P']:
                msg.data = 'print_position'
                pub.publish(msg)
                print('→ Printing position')
            elif key in ['q', 'Q', '\x03']:  # q, Q, or Ctrl+C
                print('Quitting...')
                break

    except Exception as e:
        print(f'Error: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
