import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class CustomCallback(BaseCallback):
    def __init__(self, params, dir_name, verbose=0, *args, **kwargs):
        super().__init__(verbose, *args, **kwargs)
        self.params = params
        self.dir_name = dir_name
        self.steps = 0
        self.save_freq = params['train'].get('save_every', 200)
        self.save_path = dir_name
        self.prev_end_episode = 0
        self.episode_rewards = 0
        self.episode_costs = 0
        self.episode_corrective_actions = 0
        self.episode_count = 0
        self.episode_length = 0
        self.plot = True
        self.positions = []
        self.velocities = []
        self.thetas = []
        self.actions = []
        self.record_every = params['train'].get('record_every', 100)
        self.recorded_videos = 0
        
        self.task = params['task']
        self.level = params['level']

        self.all_rewards = []
        self.all_costs = []
        self.all_corrective_actions = []
        self.all_safe_actions = []

        if self.task == 'Goal':
            self.goal_position = []
            if self.level != '0':
                self.hazard_lidars = []
                self.vase_lidars = []

        # if self.params['base'].get('wandb_enabled', False):
        #     self.video_log_step = 0
        #     self.metric_log_step = 0


    def load_previous_run(self, data, prev_end_episode):
        self.all_rewards = data['rewards']
        self.all_costs = data['costs']
        self.all_corrective_actions = data['corrective_actions']
        self.all_safe_actions = data['safe_actions']
        self.episode_count = prev_end_episode + 1
        self.prev_end_episode = prev_end_episode

        self.all_rewards = self.all_rewards[:prev_end_episode+1]

        self.all_costs = self.all_costs[:prev_end_episode+1]
        self.all_corrective_actions = self.all_corrective_actions[:prev_end_episode+1]

        if self.params['base'].get('wandb_enabled', False):
            for i in range(len(self.all_rewards)):
                wandb.log({
                    "Reward": self.all_rewards[i],
                    "Cost": self.all_costs[i],
                    "No. of Corrective Actions": self.all_corrective_actions[i],
                    "% Safe Actions": self.all_safe_actions[i]})
                # },step = self.metric_log_step)
                # self.metric_log_step += 1
        
    def _on_step(self) -> bool:
        self.steps += 1
        self._update_episode_metrics()

        if self._should_record():
            if self.locals['dones'][-1]:
                self._log_video()
                self._reset_trajectory_data()
            self._record_step()

        if self.locals['dones'][-1]:
            self._end_episode()

        return super()._on_step()

    def save_data(self):
        np.save(os.path.join(self.save_path,'episode_rewards.npy'), np.array(self.all_rewards))
        np.save(os.path.join(self.save_path,'episode_costs.npy'), np.array(self.all_costs))
        np.save(os.path.join(self.save_path,'episode_safety_calls.npy'), np.array(self.all_corrective_actions))
        np.save(os.path.join(self.save_path,'episode_percent_safe_actions.npy'), np.array(self.all_safe_actions))

        if self.params['base']['wandb_enabled']:
            run_name = self.dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")
            artifact_names = ['episode_rewards', 'episode_costs', 'episode_percent_safe_actions', 'episode_safety_calls']
            for artifact_name in artifact_names:
                artifact = wandb.Artifact(artifact_name, type="data")
                artifact.add_file(os.path.join(self.dir_name, f'{artifact_name}.npy'))
                
                artifact.metadata = {
                    "root": self.dir_name
                }
                wandb.log_artifact(artifact, aliases=[run_name])
    
    def save_model(self):
        model_name = self.params['main']['model_name']
        model_save_path = os.path.join(self.save_path, f"{model_name}_{self.episode_count}.zip")
        self.model.save(model_save_path)
        if self.verbose > 0:
            print(f"Model saved at {self.steps} steps")

        if self.params['base'].get('wandb_enabled', False):

            run_name = self.save_path.replace('/','-').replace('\\','-').replace('artifacts-', "")
            artifact = wandb.Artifact(f'{model_name}_{self.episode_count}', type="model")
            artifact.add_file(model_save_path)
            artifact.metadata = {
                "root": self.save_path
            }
            wandb.log_artifact(artifact, aliases=[run_name + f"-{self.episode_count}"])

    def _update_episode_metrics(self):
        """Update metrics for the current episode."""
        info = self.locals.get('infos', [{}])[-1]
        self.episode_rewards += info.get('reward', 0)
        self.episode_costs += info.get('cost', 0)
        cbf_optimizer_used = info.get('cbf_optimizer_used', False)
        self.episode_corrective_actions += int(cbf_optimizer_used)
        self.episode_length += 1

    def _end_episode(self):
        """End the current episode and update count."""
        self.episode_count += 1
        
    def _should_record(self) -> bool:
        """Check if the current step should be recorded."""
        return self.episode_count % self.record_every < 4 and self.episode_count > 3

    def _record_step(self):
        """Record the current step's data."""
        info = self.locals.get('infos', [{}])[-1]
        self.positions.append(info.get('position', [0, 0]))
        self.velocities.append(info.get('velocity', [0, 0]))
        self.actions.append(info.get('action', [0, 0]))
        self.thetas.append(info.get('theta', 0))

        if self.task == 'Goal':
            if self.level != '0':
                self.hazard_lidars.append(info.get('hazard_lidar', [0]*16))
                self.vase_lidars.append(info.get('vase_lidar', [0]*16))
            self.goal_position.append(info.get('goal_position', [0, 0]))

    def _log_video(self):
        """Log the trajectory video to Wandb."""
        plot_video_data = self._create_plot_video()

        if self.params['base'].get('wandb_enabled', False):
            plot_video_data = self._create_plot_video()
            if plot_video_data is None:
                return
            fps = len(plot_video_data) / 10

            wandb.log({
                f"Episode {self.episode_count-self.recorded_videos} ": wandb.Video(plot_video_data, fps=fps, format="mp4")})
            # }, step = self.video_log_step)
            # self.video_log_step += 1

            self.recorded_videos += 1
            if self.recorded_videos >= 4:
                self.recorded_videos = 0

    def _create_plot_video(self):
        """Create a video from the trajectory data."""
        plot_frames = []

        self.positions = self.positions[1:]
        self.velocities = self.velocities[1:]
        self.actions = self.actions[1:]

        if len(self.positions) == 0:
            return None
        for i, pos in enumerate(self.positions):
            fig, ax = plt.subplots()
            self._draw_plot(ax, pos, i)
            plot_frames.append(self._convert_canvas_to_image(fig))
            plt.close(fig)

        plot_video_data = np.stack(plot_frames, axis=0)
        return np.transpose(plot_video_data, (0, 3, 1, 2))

    def _draw_plot(self, ax, pos, i):
        if self.task == 'Circle':
            """Draw the plot for a single frame."""
            circle = plt.Circle((0, 0), 1.5, color='green', fill=False)
            ax.add_patch(circle)
            plt.axvline(x=1.125, color='r')
            plt.axvline(x=-1.125, color='r')
            plt.plot([p[0] for p in self.positions[:i+1]], [p[1] for p in self.positions[:i+1]], 'bo-', linewidth=0.5, markersize=2)
            plt.xlim(-2, 2)
            plt.ylim(-4, 4)
            plt.title(f'Step {i+1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax.set_aspect('equal', 'box')

            text_str = self._create_annotation_text(pos, i)
            if self.task == 'Circle':
                plt.text(2.2, 2.5, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            if self.task == 'Goal':
                plt.text(5.2, 5.2, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        if self.task == 'Goal':
            goal_pos = self.goal_position
            circle = plt.Circle((goal_pos[i][0], goal_pos[i][1]), 0.3, color = 'lightgreen', alpha=0.5, fill = True)
            ax.add_patch(circle)

            if self.level != '0':
                info = self.locals.get('infos', [{}])[-1]
                hazard_locations = info.get('hazard_positions', [[]])
                vase_locations = info.get('vase_positions', [[]])

                for x_hazard ,y_hazard, _ in hazard_locations:
                    circle = plt.Circle((x_hazard,y_hazard), 0.2, color='blue', alpha=0.5, fill = True)
                    ax.add_patch(circle)
                
                for x_vase, y_vase, _ in vase_locations:
                    circle = plt.Circle((x_vase, y_vase), 0.09, color = 'cyan', alpha=0.5, fill = True)
                    ax.add_patch(circle)

            
                x_agent, y_agent, _ = pos
                theta = self.thetas[i]
                lidar_angle = np.linspace(0, 2*np.pi, 16, endpoint=False)

                hazard_lidar = self.hazard_lidars[i]
                for hazard_lidar_value, lidar_angle in zip(hazard_lidar, lidar_angle):
                    if hazard_lidar_value == 0:         # Obstacle is not in the lidar range
                        continue
                    lidar_distance = (1-hazard_lidar_value)*3
                    x_lid = x_agent + lidar_distance*np.cos(theta + lidar_angle)
                    y_lid = y_agent + lidar_distance*np.sin(theta + lidar_angle)
                    circle = plt.Circle((x_lid,y_lid), 2*0.2, color='blue', alpha=0.5, fill = False)
                    ax.add_patch(circle)
                    plt.plot([x_agent, x_lid], [y_agent, y_lid], color='blue', linewidth=0.5, alpha=0.5)
                
                vase_lidar = self.vase_lidars[i]
                lidar_angle = np.linspace(0, 2*np.pi, 16, endpoint=False)
                for vase_lidar_value, lidar_angle in zip(vase_lidar, lidar_angle):
                    if vase_lidar_value == 0:           # Obstacle is not in the lidar range
                        continue
                    lidar_distance = (1-vase_lidar_value)*3
                    x_lid = x_agent + lidar_distance*np.cos(theta + lidar_angle)
                    y_lid = y_agent + lidar_distance*np.sin(theta + lidar_angle)
                    circle = plt.Circle((x_lid,y_lid), 2*0.09, color='cyan', alpha=0.5, fill = False)
                    ax.add_patch(circle)
                    plt.plot([x_agent, x_lid], [y_agent, y_lid], color='cyan', linewidth=0.5, alpha=0.5)
                

            plt.plot([p[0] for p in self.positions[:i+1]], [p[1] for p in self.positions[:i+1]], 'ko-', linewidth=0.5, markersize=0.5)

            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.title(f'Step {i+1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax.set_aspect('equal', 'box')

            text_str = self._create_annotation_text(pos, i)
            plt.text(2.2, 2.5, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    def _create_annotation_text(self, pos, i):
        """Create the annotation text for a single frame."""
        position_text = f"x: {pos[0]:.2f}, y: {pos[1]:.2f}"
        velocity_text = f"v_x: {self.velocities[i][0]:.2f}, v_y: {self.velocities[i][1]:.2f}"
        action_text = f"Force: {self.actions[i][0]:.2f}, Omega: {self.actions[i][1]:.2f}"
        return f"{position_text}\n{velocity_text}\n{action_text}"

    def _convert_canvas_to_image(self, fig):
        """Convert the plot canvas to an image array."""
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.tostring_rgb()
        width, height = canvas.get_width_height()
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)

    def _reset_trajectory_data(self):
        """Reset the trajectory data lists."""
        self.positions = []
        self.velocities = []
        self.actions = []
        self.thetas = []
        
        if self.task == 'Goal':
            self.goal_position = []
            if self.level != '0':
                self.hazard_lidars = []
                self.vase_lidars = []

    def _on_rollout_end(self) -> None:
        """Log episodic metrics to Wandb at the end of a rollout."""
        if self.locals.get('infos', [{}])[-1].get('episode_end', True):
            
            self.all_rewards.append(self.episode_rewards)
            self.all_costs.append(self.episode_costs)
            self.all_corrective_actions.append(self.episode_corrective_actions)
            self.all_safe_actions.append(100 - 100 * self.episode_costs / self.episode_length)

            if self.params['base'].get('wandb_enabled', False):
                wandb.log({
                    "Reward": self.episode_rewards,
                    "Cost": self.episode_costs,
                    "No. of Corrective Actions": self.episode_corrective_actions,
                    "% Safe Actions": 100 - 100 * self.episode_costs / self.episode_length})
                # }, step = self.metric_log_step)
                # self.metric_log_step += 1

            self._reset_episode_metrics()
            if self.steps == self.params['train']['total_num_steps']:
                self.save_data()
            if self.episode_count % self.save_freq == 0 or self.steps == self.params['train']['total_num_steps']:
                self.save_model()
            super()._on_rollout_end()

    def _reset_episode_metrics(self):
        """Reset the metrics for a new episode."""
        self.episode_rewards = 0
        self.episode_costs = 0
        self.episode_corrective_actions = 0
        self.episode_length = 0
