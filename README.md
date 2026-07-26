# 🏓 Neon Ball Catch (AR Edition)

A minimalist, visually polished augmented reality (AR) physics game. Control a neon-glowing paddle virtually with your index finger tip via webcam and save the falling ball!

## Prerequisites

Ensure you have a working camera/webcam connected.

## Installation

Install 
```bash
python 3.12
```

Install the required python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Game

Run the main file:

```bash
python main.py
```

## How to Play

1. **Start**: Bring your hand into the webcam frame. The game will automatically start once it detects your index finger.
2. **Move**: Move your index finger left and right to glide the neon paddle across the screen.
3. **Score**: Save the falling ball by bouncing it off the paddle. Each bounce increases your score and the speed of the ball.
4. **Game Over**: If the ball falls past the paddle, you lose.
5. **Restart**: Press `SPACE` at any point to reset/play again.
6. **Quit**: Press `ESC` to exit the game window.
