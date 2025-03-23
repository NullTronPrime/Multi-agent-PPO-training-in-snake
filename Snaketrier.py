import numpy as np
import random
import pygame
import sys
import time
import threading
import logging
import math
from numba import njit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import queue
import csv

# Enable benchmark mode for cudnn if CUDA is available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------ State Computation ------------------------ #
@njit
def compute_state(snake, food_x, food_y, grid_w, grid_h, dir_index):
    head_x, head_y = snake[0]
    state = np.zeros(16, dtype=np.float32)
    # Head & food positions (normalized)
    state[0] = head_x / grid_w
    state[1] = head_y / grid_h
    state[2] = food_x / grid_w
    state[3] = food_y / grid_h
    # Direction to food
    state[4] = (food_x - head_x) / grid_w
    state[5] = (food_y - head_y) / grid_h
    # Danger detection
    state[6] = 1.0 if (head_y - 1 < 0 or (head_x, head_y - 1) in snake[1:]) else 0.0
    state[7] = 1.0 if (head_y + 1 >= grid_h or (head_x, head_y + 1) in snake[1:]) else 0.0
    state[8] = 1.0 if (head_x - 1 < 0 or (head_x - 1, head_y) in snake[1:]) else 0.0
    state[9] = 1.0 if (head_x + 1 >= grid_w or (head_x + 1, head_y) in snake[1:]) else 0.0
    # Normalized snake length
    state[10] = len(snake) / (grid_w * grid_h)
    # One-hot current direction
    if dir_index == 0:
        state[11:15] = np.array([1, 0, 0, 0], dtype=np.float32)
    elif dir_index == 1:
        state[11:15] = np.array([0, 1, 0, 0], dtype=np.float32)
    elif dir_index == 2:
        state[11:15] = np.array([0, 0, 1, 0], dtype=np.float32)
    elif dir_index == 3:
        state[11:15] = np.array([0, 0, 0, 1], dtype=np.float32)
    # Food direction binary flag
    state[15] = 1.0 if len(snake) > 1 else 0.0
    return state

# ------------------------ Environment ------------------------ #
class SnakeEnv:
    def __init__(self, grid_w=20, grid_h=20, cell=20, snake_color=(0,255,0)):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell = cell
        self.snake_color = snake_color
        self.reset()
        
    def reset(self):
        self.snake = [(self.grid_w//2, self.grid_h//2)]
        self.direction = 3  # Start moving right
        self.spawn_food()
        self.done = False
        self.steps_without_food = 0
        self.max_steps_without_food = 2 * (self.grid_w + self.grid_h)
        return self.get_state()
        
    def spawn_food(self):
        while True:
            self.food = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if self.food not in self.snake:
                break
                
    def step(self, action):
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3
            
        if self.direction == 0:
            delta = (0, -1)
        elif self.direction == 1:
            delta = (0, 1)
        elif self.direction == 2:
            delta = (-1, 0)
        elif self.direction == 3:
            delta = (1, 0)
            
        head = self.snake[0]
        new_head = (head[0] + delta[0], head[1] + delta[1])
        
        if (new_head[0] < 0 or new_head[0] >= self.grid_w or 
            new_head[1] < 0 or new_head[1] >= self.grid_h or 
            new_head in self.snake):
            self.done = True
            reward = -10.0
            return self.get_state(), reward, self.done, {}
            
        self.snake.insert(0, new_head)
        reward = 0.0
        self.steps_without_food += 1
        
        if new_head == self.food:
            reward = 10.0 + len(self.snake) * 0.1
            self.spawn_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            curr_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if curr_dist < prev_dist else -0.1
                
        if self.steps_without_food >= self.max_steps_without_food:
            self.done = True
            reward = -5.0
            
        return self.get_state(), reward, self.done, {"score": len(self.snake)}
        
    def get_state(self):
        return compute_state(self.snake, self.food[0], self.food[1],
                             self.grid_w, self.grid_h, self.direction)
                             
    def render(self, screen, offset_x=0, offset_y=0, cell_size=None):
        if cell_size is None:
            cell_size = self.cell
        viewport_width = self.grid_w * cell_size
        viewport_height = self.grid_h * cell_size
        
        rect = pygame.Rect(offset_x, offset_y, viewport_width, viewport_height)
        pygame.draw.rect(screen, (15, 15, 25), rect)
        
        for x in range(0, viewport_width, cell_size):
            pygame.draw.line(screen, (30, 30, 40), (offset_x + x, offset_y), (offset_x + x, offset_y + viewport_height))
        for y in range(0, viewport_height, cell_size):
            pygame.draw.line(screen, (30, 30, 40), (offset_x, offset_y + y), (offset_x + viewport_width, offset_y + y))
            
        for i, cell in enumerate(self.snake):
            cell_rect = pygame.Rect(offset_x + cell[0]*cell_size, offset_y + cell[1]*cell_size, cell_size-1, cell_size-1)
            if i == 0:
                head_color = tuple(min(255, c + 50) for c in self.snake_color)
                pygame.draw.rect(screen, head_color, cell_rect)
                eye_size = cell_size // 8
                if self.direction == 0:
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + cell_size//3, offset_y + cell[1]*cell_size + cell_size//3), eye_size)
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + 2*cell_size//3, offset_y + cell[1]*cell_size + cell_size//3), eye_size)
                elif self.direction == 1:
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + cell_size//3, offset_y + cell[1]*cell_size + 2*cell_size//3), eye_size)
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + 2*cell_size//3, offset_y + cell[1]*cell_size + 2*cell_size//3), eye_size)
                elif self.direction == 2:
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + cell_size//3, offset_y + cell[1]*cell_size + cell_size//3), eye_size)
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + cell_size//3, offset_y + cell[1]*cell_size + 2*cell_size//3), eye_size)
                elif self.direction == 3:
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + 2*cell_size//3, offset_y + cell[1]*cell_size + cell_size//3), eye_size)
                    pygame.draw.circle(screen, (0,0,0), (offset_x + cell[0]*cell_size + 2*cell_size//3, offset_y + cell[1]*cell_size + 2*cell_size//3), eye_size)
            else:
                intensity = max(100, self.snake_color[1] - i * 10)
                body_color = (self.snake_color[0]//2, intensity, self.snake_color[2]//2)
                pygame.draw.rect(screen, body_color, cell_rect)
                
        food_rect = pygame.Rect(offset_x + self.food[0]*cell_size, offset_y + self.food[1]*cell_size, cell_size-1, cell_size-1)
        pygame.draw.rect(screen, (255, 50, 50), food_rect)
        pygame.draw.circle(screen, (255, 200, 200),
                           (offset_x + self.food[0]*cell_size + cell_size//4, offset_y + self.food[1]*cell_size + cell_size//4),
                           cell_size//8)

# ------------------------ PPO Network ------------------------ #
class PPOAgent(nn.Module):
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

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    def clear(self):
        self.__init__()

# ------------------------ Hyperparameters & Device ------------------------ #
state_dim = 16
action_dim = 4
lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
ppo_epochs = 10
batch_size = 128
rollout_steps = 1024
entropy_coef = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(state_dim, action_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=lr)

# ------------------------ Multi-Agent Setup ------------------------ #
NUM_AGENTS = 9
agent_colors = [
    (0,255,0), (0,180,255), (255,150,0),
    (255,0,255), (255,255,0), (0,255,255),
    (200,100,50), (100,200,50), (50,100,200)
]
envs = [SnakeEnv(grid_w=20, grid_h=20, snake_color=agent_colors[i]) for i in range(NUM_AGENTS)]
buffer = RolloutBuffer()
vis_queue = queue.Queue()
csv_queue = queue.Queue()  # Queue for CSV logging

# ------------------------ GAE & PPO Update ------------------------ #
def compute_gae(rewards, values, dones, gamma, gae_lambda):
    next_value = 0
    advantages = []
    returns = []
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1 or dones[step]:
            next_value = 0
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        advantage = delta + gamma * gae_lambda * (1 - dones[step]) * (next_value if step == len(rewards) - 1 else advantages[0])
        advantages.insert(0, advantage)
        returns.insert(0, advantage + values[step])
        next_value = values[step]
    return returns, advantages

def ppo_update():
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32).to(device)
    actions = torch.tensor(buffer.actions, dtype=torch.int64).to(device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
    values = torch.tensor(buffer.values, dtype=torch.float32).to(device).squeeze()
    
    returns, advantages = compute_gae(buffer.rewards, buffer.values, buffer.dones, gamma, gae_lambda)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0
    dataset_size = states.shape[0]
    
    for epoch in range(ppo_epochs):
        permutation = np.random.permutation(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            b_states = states[indices]
            b_actions = actions[indices]
            b_old_log_probs = old_log_probs[indices]
            b_returns = returns[indices]
            b_advantages = advantages[indices]
            
            probs, value = agent(b_states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(value.squeeze(), b_returns)
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
    
    avg_loss = total_loss / (ppo_epochs * dataset_size / batch_size)
    logging.info(f"PPO Loss: {avg_loss:.4f}")
    buffer.clear()

# ------------------------ Training Thread ------------------------ #
def train_thread():
    global training_done
    num_updates = 2000
    training_done = False
    best_reward = -float('inf')
    episode_rewards = [0 for _ in range(NUM_AGENTS)]
    episode_lengths = [0 for _ in range(NUM_AGENTS)]
    episode_count = 0
    states = [env.reset() for env in envs]
    
    for update in range(num_updates):
        update_start_time = time.time()
        for step in range(rollout_steps):
            for i, env in enumerate(envs):
                state = states[i]
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                with torch.no_grad():
                    probs, value = agent(state_tensor)
                dist = Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
                
                next_state, reward, done, info = env.step(action)
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                buffer.states.append(state)
                buffer.actions.append(action)
                buffer.log_probs.append(log_prob)
                buffer.rewards.append(reward)
                buffer.dones.append(done)
                buffer.values.append(value.item())
                
                states[i] = next_state
                
                if step % 10 == 0:
                    try:
                        vis_queue.put_nowait(("step", {
                            "agent_id": i,
                            "env": env,
                            "action": action,
                            "reward": reward,
                            "done": done,
                            "update": update,
                            "step": step,
                            "score": len(env.snake)
                        }))
                    except queue.Full:
                        pass
                
                if done or (len(env.snake) >= env.grid_w * env.grid_h):
                    episode_count += 1
                    if episode_rewards[i] > best_reward:
                        best_reward = episode_rewards[i]
                    if episode_count % 10 == 0:
                        logging.info(f"Episode {episode_count} (Agent {i}) - Reward: {episode_rewards[i]:.2f}, Length: {episode_lengths[i]}")
                    states[i] = env.reset()
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    if len(env.snake) >= env.grid_w * env.grid_h:
                        training_done = True
                        break
            if training_done:
                break
        
        ppo_update()
        update_duration = time.time() - update_start_time
        # Compute average reward for this update (if any nonzero)
        valid_rewards = [r for r in episode_rewards if r != 0]
        avg_reward = sum(valid_rewards)/len(valid_rewards) if valid_rewards else 0
        log_msg = f"Update {update+1} - Avg Reward: {avg_reward:.2f}, Duration: {update_duration:.2f}s, Best Reward: {best_reward:.2f}"
        logging.info(log_msg)
        try:
            vis_queue.put_nowait(("update", {
                "update": update,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "episode_count": episode_count
            }))
        except queue.Full:
            pass
        
        # Send CSV update details to CSV thread
        try:
            csv_queue.put_nowait({
                "update": update,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "update_duration": update_duration,
                "episode_count": episode_count
            })
        except queue.Full:
            pass
        
        if (update + 1) % 25 == 0:
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'best_reward': best_reward
            }, f'snake_model_update_{update+1}.pt')
            
        if training_done:
            break
    
    training_done = True
    logging.info("Training finished. Best reward: %.2f", best_reward)
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_reward': best_reward
    }, 'snake_model_final.pt')
    vis_queue.put(("done", None))
    csv_queue.put("done")

# ------------------------ Visualization Thread ------------------------ #
def visualize_thread():
    pygame.init()
    info_obj = pygame.display.Info()
    screen_width = info_obj.current_w
    screen_height = info_obj.current_h
    sidebar_width = 200  # Fixed sidebar width
    available_width = screen_width - sidebar_width
    cols = math.ceil(math.sqrt(NUM_AGENTS))
    rows = math.ceil(NUM_AGENTS / cols)
    viewport_width = available_width // cols
    viewport_height = screen_height // rows
    
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Multi-Agent Snake PPO")
    clock = pygame.time.Clock()
    
    title_font = pygame.font.SysFont('Arial', 20, bold=True)
    font = pygame.font.SysFont('Arial', 16)
    small_font = pygame.font.SysFont('Arial', 12)
    
    current_update = 0
    avg_reward = 0
    best_reward = 0
    episode_count = 0
    last_actions = ["None" for _ in range(NUM_AGENTS)]
    demo_envs = envs.copy()
    
    running = True
    paused = False
    fps = 15
    visualize_every = 2
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fps = min(60, fps + 5)
                elif event.key == pygame.K_DOWN:
                    fps = max(5, fps - 5)
                    
        try:
            msg_type, data = vis_queue.get_nowait()
            if msg_type == "update":
                current_update = data["update"]
                avg_reward = data["avg_reward"]
                best_reward = data["best_reward"]
                episode_count = data["episode_count"]
            elif msg_type == "step":
                agent_id = data["agent_id"]
                demo_envs[agent_id] = data["env"]
                last_actions[agent_id] = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT"}[data["action"]]
            elif msg_type == "done":
                running = False
        except queue.Empty:
            pass
        
        frame_count += 1
        if frame_count % visualize_every != 0 and not paused:
            clock.tick(fps * visualize_every)
            continue
        
        screen.fill((30, 30, 40))
        for i, env in enumerate(demo_envs):
            col = i % cols
            row = i // cols
            offset_x = col * viewport_width
            offset_y = row * viewport_height
            cell_size = int(min(viewport_width / env.grid_w, viewport_height / env.grid_h))
            env.render(screen, offset_x=offset_x, offset_y=offset_y, cell_size=cell_size)
            overlay_rect = pygame.Rect(offset_x, offset_y, viewport_width, 30)
            pygame.draw.rect(screen, (40,40,50), overlay_rect)
            info_text = font.render(f"Agent {i} | Score: {len(env.snake)} | Last Action: {last_actions[i]}", True, (220,220,220))
            screen.blit(info_text, (offset_x + 10, offset_y + 5))
        
        sidebar_rect = pygame.Rect(screen_width - sidebar_width, 0, sidebar_width, screen_height)
        pygame.draw.rect(screen, (40, 40, 50), sidebar_rect)
        title = title_font.render("Multi-Agent PPO", True, (255, 255, 255))
        screen.blit(title, (screen_width - sidebar_width + 10, 10))
        y_pos = 50
        texts = [
            f"Update: {current_update + 1}",
            f"Episodes: {episode_count}",
            f"Avg Reward: {avg_reward:.2f}",
            f"Best Reward: {best_reward:.2f}",
            f"FPS: {fps}"
        ]
        for text in texts:
            text_surface = font.render(text, True, (220, 220, 220))
            screen.blit(text_surface, (screen_width - sidebar_width + 10, y_pos))
            y_pos += 25
        y_pos += 20
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "UP - Increase speed",
            "DOWN - Decrease speed"
        ]
        for text in controls:
            text_surface = small_font.render(text, True, (180, 180, 180))
            screen.blit(text_surface, (screen_width - sidebar_width + 10, y_pos))
            y_pos += 20
        pygame.display.flip()
        
        if not paused:
            for i, env in enumerate(demo_envs):
                if not env.done:
                    state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        probs, _ = agent(state_tensor)
                    action = Categorical(probs).sample().item()
                    last_actions[i] = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT"}[action]
                    state, _, done, _ = env.step(action)
                    if done:
                        env.reset()
            clock.tick(fps)
    
    pygame.quit()

# ------------------------ CSV Logging Thread ------------------------ #
def csv_thread():
    # Open CSV file and write header
    with open("training_details.csv", mode="w", newline="") as csvfile:
        fieldnames = ["update", "avg_reward", "best_reward", "update_duration", "episode_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            try:
                msg = csv_queue.get(timeout=1)
                if msg == "done":
                    break
                writer.writerow(msg)
                csvfile.flush()
            except queue.Empty:
                continue

# ------------------------ Main ------------------------ #
if __name__ == "__main__":
    t1 = threading.Thread(target=train_thread)
    t2 = threading.Thread(target=visualize_thread)
    t3 = threading.Thread(target=csv_thread)
    
    t1.start()
    t2.start()
    t3.start()
    
    t1.join()
    t2.join()
    t3.join()
