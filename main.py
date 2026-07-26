import cv2
import mediapipe as mp

print(hasattr(mp, "solutions"))

import numpy as np
import math
import random
import time

W, H = 1280, 720
PADDLE_W = 160
PADDLE_H = 14
PADDLE_Y = H - 90
BALL_R = 18
BALL_SPEED_0 = 18
SPEED_INC = 0.5

COL_PADDLE = (0, 220, 255)
COL_BALL = (80, 255, 160)
COL_TRAIL = (40, 140,  90)
COL_SCORE = (255, 230, 100)
COL_DEAD = (50,  60, 255)

FONT = cv2.FONT_HERSHEY_DUPLEX

def lerp(a, b, t):
    return a + (b - a) * t

def draw_glow_circle(img, cx, cy, r, color, layers=4, base_alpha=0.35):
    for i in range(layers, 0, -1):
        radius = r + i * 6
        alpha = base_alpha / i
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_glow_rect(img, x, y, w, h, color, layers=3, base_alpha=0.4):
    r = h // 2
    for i in range(layers, 0, -1):
        ex = x - i * 4
        ey = y - i * 4
        ew = w + i * 8
        eh = h + i * 8
        alpha = base_alpha / i
        overlay = img.copy()
        cv2.rectangle(overlay, (ex + r, ey), (ex + ew - r, ey + eh), color, -1)
        cv2.rectangle(overlay, (ex, ey + r), (ex + ew, ey + eh - r), color, -1)
        cv2.circle(overlay, (ex + r, ey + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + ew - r, ey + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + r, ey + eh - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (ex + ew - r, ey + eh - r), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(img, (x + r, y + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - r, y + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + r, y + h - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - r, y + h - r), r, color, -1, cv2.LINE_AA)

    shine_color = tuple(min(255, c + 90) for c in color)
    sy = y + 2
    sh = max(3, h // 3)
    cv2.rectangle(img, (x + r, sy), (x + w - r, sy + sh), shine_color, -1)
    cv2.addWeighted(img, 0.4, img.copy(), 0.6, 0, img)

class Spark:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 9)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 4
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.10)
        self.r = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.35
        self.life -= self.decay
        return self.life > 0

    def draw(self, img):
        alpha = self.life
        c = tuple(int(ch * alpha) for ch in self.color)
        cv2.circle(img, (int(self.x), int(self.y)), self.r, c, -1, cv2.LINE_AA)

class BallCatchGame:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.60,
        )
        self._reset()
        self.paddle_x = W // 2 - PADDLE_W // 2
        self.paddle_x_tgt = self.paddle_x
        self.finger_x = W // 2
        self.finger_y = H // 2
        self.hand_visible = False
        self.state = "waiting"
        self.cam_frame = np.zeros((H, W, 3), dtype=np.uint8)

    def _reset(self):
        self.score = 0
        self.ball_x = float(W // 2)
        self.ball_y = float(H // 4)
        speed = BALL_SPEED_0
        angle = random.uniform(math.pi * 0.3, math.pi * 0.7)
        self.ball_vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.ball_vy = math.sin(angle) * speed
        self.speed = speed
        self.sparks = []
        self.trail = []
        self.shake = 0

    def _process_hand(self, rgb):
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.hand_visible = False
            return
        lm = res.multi_hand_landmarks[0].landmark
        self.finger_x = int(lm[8].x * W)
        self.finger_y = int(lm[8].y * H)
        self.hand_visible = True

    def _update(self):
        if self.state != "playing":
            return

        self.paddle_x_tgt = self.finger_x - PADDLE_W // 2
        self.paddle_x_tgt = max(0, min(W - PADDLE_W, self.paddle_x_tgt))
        self.paddle_x += (self.paddle_x_tgt - self.paddle_x) * 0.30

        px = int(self.paddle_x)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        self.trail.append((int(self.ball_x), int(self.ball_y), 1.0))
        self.trail = [(x, y, a - 0.07) for x, y, a in self.trail if a > 0]

        if self.ball_x - BALL_R <= 0:
            self.ball_x = BALL_R
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + BALL_R >= W:
            self.ball_x = W - BALL_R
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y - BALL_R <= 0:
            self.ball_y = BALL_R
            self.ball_vy = abs(self.ball_vy)

        ball_bottom = self.ball_y + BALL_R
        paddle_top = PADDLE_Y
        paddle_left = px
        paddle_right = px + PADDLE_W

        if (paddle_top - 6 <= ball_bottom <= paddle_top + 12 and
                paddle_left - BALL_R <= self.ball_x <= paddle_right + BALL_R and
                self.ball_vy > 0):
            hit_pos = (self.ball_x - (paddle_left + PADDLE_W / 2)) / (PADDLE_W / 2)
            self.ball_vy = -abs(self.ball_vy)
            self.ball_vx += hit_pos * 3.5
            max_speed = self.speed + SPEED_INC * (self.score + 1)
            total = math.hypot(self.ball_vx, self.ball_vy)
            if total > 0:
                self.ball_vx = self.ball_vx / total * max_speed
                self.ball_vy = self.ball_vy / total * max_speed

            self.ball_y = PADDLE_Y - BALL_R - 1
            self.score += 1
            self.speed = BALL_SPEED_0 + SPEED_INC * self.score

            for _ in range(20):
                self.sparks.append(Spark(self.ball_x, self.ball_y, COL_BALL))
            self.shake = 6

        if self.ball_y - BALL_R > H + 20:
            self.state = "dead"
            self.shake = 15
            for _ in range(40):
                self.sparks.append(Spark(W // 2, H // 2, COL_DEAD))

        self.sparks = [s for s in self.sparks if s.update()]

        if self.shake > 0:
            self.shake -= 1

    def _draw(self, frame):
        t = time.time()
        ox = random.randint(-self.shake, self.shake) if self.shake else 0
        oy = random.randint(-self.shake, self.shake) if self.shake else 0

        frame[:] = np.roll(self.cam_frame, (ox, oy), axis=(1, 0))

        for tx, ty, alpha in self.trail:
            r = max(2, int(BALL_R * 0.5 * alpha))
            c = tuple(int(ch * alpha * 0.7) for ch in COL_TRAIL)
            cv2.circle(frame, (tx + ox, ty + oy), r, c, -1, cv2.LINE_AA)

        for s in self.sparks:
            s.draw(frame)

        px = int(self.paddle_x) + ox
        draw_glow_rect(frame, px, PADDLE_Y + oy, PADDLE_W, PADDLE_H, COL_PADDLE)

        bx, by = int(self.ball_x) + ox, int(self.ball_y) + oy
        if self.state != "dead":
            draw_glow_circle(frame, bx, by, BALL_R, COL_BALL)
            cv2.circle(frame, (bx, by), BALL_R, COL_BALL, -1, cv2.LINE_AA)
            cv2.circle(frame, (bx - 5, by - 5), BALL_R // 3, (220, 255, 230), -1, cv2.LINE_AA)

        if self.hand_visible:
            fx, fy = self.finger_x, self.finger_y
            cv2.circle(frame, (fx, fy), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy),  4, (255, 255, 255), -1, cv2.LINE_AA)

        danger_alpha = 0.18 + 0.12 * abs(math.sin(t * 4))
        ov = frame.copy()
        cv2.line(ov, (0, PADDLE_Y + PADDLE_H + 30), (W, PADDLE_Y + PADDLE_H + 30), (50, 60, 255), 2)
        cv2.addWeighted(ov, danger_alpha, frame, 1 - danger_alpha, 0, frame)

        score_txt = str(self.score)
        cv2.putText(frame, score_txt, (28, 60), FONT, 1.8, (30, 40, 60), 5, cv2.LINE_AA)
        cv2.putText(frame, score_txt, (28, 60), FONT, 1.8, COL_SCORE, 3, cv2.LINE_AA)
        cv2.putText(frame, "SCORE", (30, 80), FONT, 0.38, (140, 160, 200), 1, cv2.LINE_AA)

        if self.state == "waiting":
            self._draw_waiting(frame, t)
        elif self.state == "dead":
            self._draw_dead(frame, t)

    def _draw_waiting(self, frame, t):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (8, 4, 18), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        pulse = int(abs(math.sin(t * 2)) * 25)
        title = "BALL CATCH"
        tw = cv2.getTextSize(title, FONT, 2.8, 5)[0][0]
        for thickness, alpha_val in [(18, 0.05), (10, 0.12), (5, 0.3)]:
            ov2 = frame.copy()
            cv2.putText(ov2, title, (W // 2 - tw // 2, H // 2 - 60), FONT, 2.8, COL_PADDLE, thickness, cv2.LINE_AA)
            cv2.addWeighted(ov2, alpha_val, frame, 1 - alpha_val, 0, frame)
        cv2.putText(frame, title, (W // 2 - tw // 2, H // 2 - 60), FONT, 2.8, COL_PADDLE, 4, cv2.LINE_AA)

        sub = "Move your INDEX FINGER to begin"
        sw = cv2.getTextSize(sub, FONT, 0.75, 1)[0][0]
        cv2.putText(frame, sub, (W // 2 - sw // 2, H // 2 + 30), FONT, 0.75, (200, 220, 255), 1, cv2.LINE_AA)

        hint = "Keep the ball from falling!"
        hw = cv2.getTextSize(hint, FONT, 0.58, 1)[0][0]
        cv2.putText(frame, hint, (W // 2 - hw // 2, H // 2 + 75), FONT, 0.58, (int(120 + pulse), int(150 + pulse), int(200 + pulse)), 1, cv2.LINE_AA)

    def _draw_dead(self, frame, t):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (5, 3, 15), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        pulse = int(abs(math.sin(t * 3)) * 30)
        go_txt = "GAME OVER"
        tw = cv2.getTextSize(go_txt, FONT, 2.6, 5)[0][0]
        go_col = (int(80 + pulse), int(70 + pulse), 255)
        cv2.putText(frame, go_txt, (W // 2 - tw // 2, H // 2 - 60), FONT, 2.6, go_col, 5, cv2.LINE_AA)

        sc_txt = f"You saved the ball  {self.score}  time{'s' if self.score != 1 else ''}!"
        sw = cv2.getTextSize(sc_txt, FONT, 0.80, 1)[0][0]
        cv2.putText(frame, sc_txt, (W // 2 - sw // 2, H // 2 + 20), FONT, 0.80, (200, 220, 255), 2, cv2.LINE_AA)

        restart = "Press  SPACE  to play again   |   ESC  to quit"
        rw = cv2.getTextSize(restart, FONT, 0.60, 1)[0][0]
        cv2.putText(frame, restart, (W // 2 - rw // 2, H // 2 + 80), FONT, 0.60, (140, 160, 200), 1, cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        cv2.namedWindow("Ball Catch", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Catch", W, H)

        frame_out = np.zeros((H, W, 3), dtype=np.uint8)

        while True:
            ret, cam = cap.read()
            if not ret:
                cam = np.zeros((H, W, 3), dtype=np.uint8)

            cam = cv2.resize(cam, (W, H))
            cam = cv2.flip(cam, 1)
            self.cam_frame = cam.copy()

            rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            self._process_hand(rgb)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(' '):
                self._reset()
                self.state = "playing"

            if self.state == "waiting" and self.hand_visible:
                self._reset()
                self.state = "playing"

            self._update()
            self._draw(frame_out)

            cv2.imshow("Ball Catch", frame_out)

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    game = BallCatchGame()
    game.run()