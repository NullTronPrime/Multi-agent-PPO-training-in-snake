import pygame
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import os
import time

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# PPO Network
# =============================================================================
class PPOAgent(nn.Module):
    """
    A PPO agent network with a shared backbone, actor head, and critic head.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPOAgent, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

# =============================================================================
# State Computation Function
# =============================================================================
def compute_state(snake, food_x, food_y, grid_w, grid_h, dir_index):
    """
    Compute a normalized state vector for the snake game.
    """
    head_x, head_y = snake[0]
    state = np.zeros(16, dtype=np.float32)
    # Normalize head and food positions
    state[0] = head_x / grid_w
    state[1] = head_y / grid_h
    state[2] = food_x / grid_w
    state[3] = food_y / grid_h
    # Relative food direction
    state[4] = (food_x - head_x) / grid_w
    state[5] = (food_y - head_y) / grid_h
    # Danger detection in 4 directions
    state[6] = 1.0 if (head_y - 1 < 0 or (head_x, head_y - 1) in snake[1:]) else 0.0
    state[7] = 1.0 if (head_y + 1 >= grid_h or (head_x, head_y + 1) in snake[1:]) else 0.0
    state[8] = 1.0 if (head_x - 1 < 0 or (head_x - 1, head_y) in snake[1:]) else 0.0
    state[9] = 1.0 if (head_x + 1 >= grid_w or (head_x + 1, head_y) in snake[1:]) else 0.0
    # Snake length normalized by grid size
    state[10] = len(snake) / (grid_w * grid_h)
    # One-hot encoding for current direction
    if dir_index == 0:
        state[11:15] = np.array([1, 0, 0, 0], dtype=np.float32)
    elif dir_index == 1:
        state[11:15] = np.array([0, 1, 0, 0], dtype=np.float32)
    elif dir_index == 2:
        state[11:15] = np.array([0, 0, 1, 0], dtype=np.float32)
    elif dir_index == 3:
        state[11:15] = np.array([0, 0, 0, 1], dtype=np.float32)
    # Binary flag for food direction
    state[15] = 1.0 if len(snake) > 1 else 0.0
    return state

# =============================================================================
# Snake Game Class
# =============================================================================
class SnakeGame:
    def __init__(self, grid_w=20, grid_h=20, cell_size=30):
        """
        Initialize the game settings, pygame window, fonts, load the model, and reset game state.
        """
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.width = grid_w * cell_size
        self.height = grid_h * cell_size
        self.score_height = 50
        self.sidebar_width = 300
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width + self.sidebar_width, self.height + self.score_height))
        pygame.display.set_caption("AI Snake Game")
        self.clock = pygame.time.Clock()
        
        # Load fonts
        self.title_font = pygame.font.SysFont('Arial', 28, bold=True)
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Load the trained AI model
        self.load_model()
        
        # Initialize game state
        self.reset_game()
        
        # Colors
        self.BG_COLOR = (15, 15, 30)
        self.GRID_COLOR = (30, 30, 50)
        self.SCORE_BG = (40, 40, 60)
        self.SIDEBAR_BG = (30, 30, 45)
        self.TEXT_COLOR = (220, 220, 220)
        self.SNAKE_HEAD = (50, 255, 120)
        self.SNAKE_BODY = (0, 200, 80)
        self.FOOD_COLOR = (255, 50, 50)
        self.FOOD_SHINE = (255, 200, 200)
        
        # Control flags and settings
        self.paused = False
        self.game_over = False
        self.player_mode = False
        self.human_action = None
        self.speed = 10  # Frames per second
        
        # Statistics
        self.games_played = 0
        self.high_score = 0
        self.total_score = 0
        self.avg_score = 0
        self.game_stats = []
    
    def load_model(self):
        """
        Loads the trained model from disk.
        """
        # Find the final model or the latest update model
        model_files = [f for f in os.listdir('.') if f.startswith('snake_model_') and f.endswith('.pt')]
        if 'snake_model_final.pt' in model_files:
            model_path = 'snake_model_final.pt'
        else:
            update_models = [f for f in model_files if 'update_' in f]
            if update_models:
                update_numbers = [int(f.split('update_')[1].split('.pt')[0]) for f in update_models]
                max_update = max(update_numbers)
                model_path = f'snake_model_update_{max_update}.pt'
            else:
                raise FileNotFoundError("No trained model found. Train a model first.")
        
        print(f"Loading model from {model_path}")
        
        self.state_dim = 16
        self.action_dim = 4
        self.agent = PPOAgent(self.state_dim, self.action_dim).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.eval()  # Set model to evaluation mode
        
        best_reward = checkpoint.get('best_reward', 'Unknown')
        print(f"Model loaded - Best recorded reward: {best_reward}")
    
    def reset_game(self):
        """
        Resets the game state.
        """
        self.snake = [(self.grid_w // 2, self.grid_h // 2)]
        self.direction = 3  # Start moving right (0: Up, 1: Down, 2: Left, 3: Right)
        self.spawn_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_start_time = time.time()
    
    def spawn_food(self):
        """
        Places food randomly on the grid, ensuring it doesn't appear on the snake.
        """
        while True:
            self.food = (np.random.randint(0, self.grid_w), np.random.randint(0, self.grid_h))
            if self.food not in self.snake:
                break
    
    def get_state(self):
        """
        Returns the current game state as a normalized vector.
        """
        return compute_state(self.snake, self.food[0], self.food[1],
                             self.grid_w, self.grid_h, self.direction)
    
    def get_ai_action(self):
        """
        Uses the AI model to decide on an action.
        Adds a batch dimension so the output has shape (1, 4).
        """
        state_tensor = torch.tensor(self.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = self.agent(state_tensor)
        dist = Categorical(probs)
        action = dist.sample().item()
        return action, probs.cpu().numpy()  # probs now has shape (1, 4)
    
    def update(self):
        """
        Updates the game state by moving the snake according to the selected action.
        """
        if self.paused or self.game_over:
            return
        
        # Determine action (from AI or human)
        if self.player_mode:
            action = self.human_action
            self.human_action = None
            if action is None:
                return  # Wait for human input
        else:
            action, _ = self.get_ai_action()
        
        # Update direction ensuring no reverse movement
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3
        
        # Determine new head position
        head_x, head_y = self.snake[0]
        if self.direction == 0:      # UP
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:    # DOWN
            new_head = (head_x, head_y + 1)
        elif self.direction == 2:    # LEFT
            new_head = (head_x - 1, head_y)
        elif self.direction == 3:    # RIGHT
            new_head = (head_x + 1, head_y)
        
        # Check for collisions with wall or self
        if (new_head[0] < 0 or new_head[0] >= self.grid_w or
            new_head[1] < 0 or new_head[1] >= self.grid_h or
            new_head in self.snake):
            self.game_over = True
            self.games_played += 1
            self.total_score += self.score
            self.avg_score = self.total_score / self.games_played
            self.high_score = max(self.high_score, self.score)
            self.game_stats.append({
                'score': self.score,
                'steps': self.steps,
                'time': time.time() - self.game_start_time
            })
            return
        
        # Move the snake (insert new head)
        self.snake.insert(0, new_head)
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()  # Remove tail
        
        self.steps += 1
    
    def draw_snake_segment(self, pos, segment_type='body', direction=None):
        """
        Draws a snake segment (head or body).
        """
        x, y = pos
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        if segment_type == 'head':
            color = self.SNAKE_HEAD
            pygame.draw.rect(self.screen, color, rect, border_radius=self.cell_size // 4)
            # Draw eyes
            eye_size = self.cell_size // 8
            eye_offset = self.cell_size // 4
            if direction == 0:  # UP
                left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + eye_offset)
                right_eye = (x * self.cell_size + self.cell_size - eye_offset - eye_size, y * self.cell_size + eye_offset)
            elif direction == 1:  # DOWN
                left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + self.cell_size - eye_offset - eye_size)
                right_eye = (x * self.cell_size + self.cell_size - eye_offset - eye_size, y * self.cell_size + self.cell_size - eye_offset - eye_size)
            elif direction == 2:  # LEFT
                left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + eye_offset)
                right_eye = (x * self.cell_size + eye_offset, y * self.cell_size + self.cell_size - eye_offset - eye_size)
            else:  # RIGHT
                left_eye = (x * self.cell_size + self.cell_size - eye_offset - eye_size, y * self.cell_size + eye_offset)
                right_eye = (x * self.cell_size + self.cell_size - eye_offset - eye_size, y * self.cell_size + self.cell_size - eye_offset - eye_size)
            pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(left_eye[0], left_eye[1], eye_size, eye_size))
            pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(right_eye[0], right_eye[1], eye_size, eye_size))
        else:
            intensity = max(100, 220 - len(self.snake) + self.snake.index(pos))
            color = (self.SNAKE_BODY[0], intensity, self.SNAKE_BODY[2])
            pygame.draw.rect(self.screen, color, rect, border_radius=self.cell_size // 5)
    
    def draw_food(self):
        """
        Draws the food with a shine effect.
        """
        x, y = self.food
        food_rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.FOOD_COLOR, food_rect, border_radius=self.cell_size // 3)
        shine_size = self.cell_size // 4
        shine_pos = (x * self.cell_size + self.cell_size // 5, y * self.cell_size + self.cell_size // 5)
        pygame.draw.circle(self.screen, self.FOOD_SHINE, shine_pos, shine_size // 2)
    
    def draw_grid(self):
        """
        Draws the grid lines for the game area.
        """
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))
    
    def draw_score_panel(self):
        """
        Draws the score panel below the game area.
        """
        score_rect = pygame.Rect(0, self.height, self.width + self.sidebar_width, self.score_height)
        pygame.draw.rect(self.screen, self.SCORE_BG, score_rect)
        score_text = self.font.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (20, self.height + 15))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.TEXT_COLOR)
        self.screen.blit(steps_text, (150, self.height + 15))
        mode_text = self.font.render(f"Mode: {'Human' if self.player_mode else 'AI'}", True, self.TEXT_COLOR)
        self.screen.blit(mode_text, (280, self.height + 15))
        if self.paused:
            paused_text = self.font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(paused_text, (420, self.height + 15))
        if self.game_over:
            gameover_text = self.font.render("GAME OVER - Press R to Restart", True, (255, 100, 100))
            self.screen.blit(gameover_text, (480, self.height + 15))
    
    def draw_sidebar(self):
        """
        Draws the sidebar with game statistics and AI decision info.
        """
        sidebar_rect = pygame.Rect(self.width, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, self.SIDEBAR_BG, sidebar_rect)
        title = self.title_font.render("AI Snake Game", True, self.TEXT_COLOR)
        self.screen.blit(title, (self.width + 20, 20))
        y_pos = 70
        stats = [
            f"High Score: {self.high_score}",
            f"Games Played: {self.games_played}",
            f"Average Score: {self.avg_score:.2f}",
            f"Speed: {self.speed} FPS"
        ]
        for text in stats:
            text_surface = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(text_surface, (self.width + 20, y_pos))
            y_pos += 30
        y_pos += 20
        controls_title = self.font.render("Controls:", True, (180, 180, 180))
        self.screen.blit(controls_title, (self.width + 20, y_pos))
        y_pos += 30
        controls = [
            "Space - Pause/Resume",
            "R - Restart game",
            "M - Toggle AI/Human mode",
            "+/- - Change speed",
            "",
            "Arrow keys - Control snake",
            "(in Human mode)"
        ]
        for text in controls:
            text_surface = self.small_font.render(text, True, (160, 160, 160))
            self.screen.blit(text_surface, (self.width + 30, y_pos))
            y_pos += 25
        
        # AI Decision probabilities
        if not self.player_mode and not self.game_over:
            y_pos += 20
            prob_title = self.font.render("AI Decision:", True, (180, 180, 180))
            self.screen.blit(prob_title, (self.width + 20, y_pos))
            y_pos += 30
            # Get AI action probabilities with batch dimension
            _, probs = self.get_ai_action()
            directions = ["Up", "Down", "Left", "Right"]
            max_prob = max(probs[0])
            for i, (direction, prob) in enumerate(zip(directions, probs[0])):
                prob_text = self.small_font.render(f"{direction}: {prob:.2f}", True,
                                                   (255, 255, 255) if prob == max_prob else (160, 160, 160))
                self.screen.blit(prob_text, (self.width + 30, y_pos))
                bar_width = int(150 * prob)
                bar_color = (50, 200, 50) if prob == max_prob else (100, 100, 100)
                pygame.draw.rect(self.screen, bar_color, pygame.Rect(self.width + 120, y_pos + 5, bar_width, 10))
                y_pos += 25
    
    def draw(self):
        """
        Draws the entire game screen.
        """
        self.screen.fill(self.BG_COLOR)
        self.draw_grid()
        self.draw_food()
        for i, segment in enumerate(self.snake):
            if i == 0:
                self.draw_snake_segment(segment, 'head', self.direction)
            else:
                self.draw_snake_segment(segment, 'body')
        self.draw_score_panel()
        self.draw_sidebar()
        pygame.display.flip()
    
    def handle_input(self):
        """
        Processes user input.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_m:
                    self.player_mode = not self.player_mode
                    print(f"Switched to {'Human' if self.player_mode else 'AI'} mode")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.speed = min(60, self.speed + 5)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed - 5)
                
                # Human control if in player mode
                if self.player_mode:
                    if event.key == pygame.K_UP and self.direction != 1:
                        self.human_action = 0
                    elif event.key == pygame.K_DOWN and self.direction != 0:
                        self.human_action = 1
                    elif event.key == pygame.K_LEFT and self.direction != 3:
                        self.human_action = 2
                    elif event.key == pygame.K_RIGHT and self.direction != 2:
                        self.human_action = 3
    
    def run(self):
        """
        Main game loop.
        """
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(self.speed)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    game = SnakeGame()
    game.run()
