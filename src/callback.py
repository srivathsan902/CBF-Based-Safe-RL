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
        self.save_freq = params['train'].get('save_every', 100000)
        self.save_path = dir_name
        self.prev_end_step = 0
        self.total_num_steps = params['train']['total_num_steps'] + 5*params['train']['sliding_window']
        self.sliding_window = params['train'].get('sliding_window', 1000)
        self.episode_count = 0
        self.episode_length = 0

        self.plot = True
        self.positions = []
        self.velocities = []
        self.thetas = []
        self.actions = []
        self.record_every = params['train'].get('record_every', 50000)
        self.record_every = self.record_every // params['train']['max_steps_per_episode']
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


    def load_previous_run(self, data, prev_end_step):
        self.all_rewards = data['rewards']
        self.all_costs = data['costs']
        self.all_corrective_actions = data['corrective_actions']
        self.all_safe_actions = data['safe_actions']
        self.steps = prev_end_step + 1
        self.prev_end_step = prev_end_step

        self.all_rewards = self.all_rewards[:prev_end_step+1]

        self.all_costs = self.all_costs[:prev_end_step+1]
        self.all_corrective_actions = self.all_corrective_actions[:prev_end_step+1]

        if self.params['base'].get('wandb_enabled', False):
            for i in range(len(self.all_rewards)):
                wandb.log({
                    "Reward": sum(self.all_rewards[i-self.sliding_window:i]),
                    "Cost": sum(self.all_costs[i-self.sliding_window:i]),
                    "No. of Corrective Actions": sum(self.all_corrective_actions[i-self.sliding_window:i]),
                    "% Safe Actions": sum(self.all_safe_actions[i-self.sliding_window:i])/self.sliding_window})
                # },step = self.metric_log_step)
                # self.metric_log_step += 1
        
    def _on_step(self) -> bool:
        self.steps += 1
        self._update_metrics()
        self._log_metrics()

        if self._should_record():
            if self.locals['dones'][-1]:
                self._log_video()
                self._reset_trajectory_data()
            self._record_step()

        if self.locals['dones'][-1]:
            self._end_episode()

        return super()._on_step()

    def save_data(self):
        np.save(os.path.join(self.save_path,'rewards.npy'), np.array(self.all_rewards))
        np.save(os.path.join(self.save_path,'costs.npy'), np.array(self.all_costs))
        np.save(os.path.join(self.save_path,'safety_calls.npy'), np.array(self.all_corrective_actions))
        np.save(os.path.join(self.save_path,'percent_safe_actions.npy'), np.array(self.all_safe_actions))

        if self.params['base']['wandb_enabled']:
            run_name = self.dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")
            artifact_names = ['rewards', 'costs', 'percent_safe_actions', 'safety_calls']
            for artifact_name in artifact_names:
                artifact = wandb.Artifact(artifact_name, type="data")
                artifact.add_file(os.path.join(self.dir_name, f'{artifact_name}.npy'))
                
                artifact.metadata = {
                    "root": self.dir_name
                }
                wandb.log_artifact(artifact, aliases=[run_name])
    
    def save_model(self):
        model_name = self.params['main']['model_name']
        model_save_path = os.path.join(self.save_path, f"{model_name}_{self.episode_count+1}.zip")
        self.model.save(model_save_path)
        if self.verbose > 0:
            print(f"Model saved at {self.steps} steps")

        if self.params['base'].get('wandb_enabled', False):

            run_name = self.save_path.replace('/','-').replace('\\','-').replace('artifacts-', "")
            artifact = wandb.Artifact(f'{model_name}_{self.episode_count+1}', type="model")
            artifact.add_file(model_save_path)
            artifact.metadata = {
                "root": self.save_path
            }
            wandb.log_artifact(artifact, aliases=[run_name + f"-{self.episode_count+1}"])

    def _update_metrics(self):
        """Update metrics for the current episode."""
        info = self.locals.get('infos', [{}])[-1]
        self.all_rewards.append(info.get('reward', 0))
        self.all_costs.append(info.get('cost', 0))
        cbf_optimizer_used = info.get('cbf_optimizer_used', False)
        self.all_corrective_actions.append(int(cbf_optimizer_used))
        self.all_safe_actions.append(100 - 100*info.get('cost', 0))
        self.episode_length += 1

    def _end_episode(self):
        """End the current episode and update count."""
        self.episode_count += 1
        
    def _should_record(self) -> bool:
        """Check if the current step should be recorded."""
        if self.task == 'Goal':
            return self.episode_count % self.record_every < 2 and self.episode_count > 3
        else:
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
            plt.plot([p[0] for p in self.positions[:i+1]], [p[1] for p in self.positions[:i+1]], 'bo-', linewidth=0.5, markersize=1)
            plt.xlim(-2, 2)
            plt.ylim(-4, 4)
            plt.title(f'Step {i+1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            ax.set_aspect('equal', 'box')

            text_str = self._create_annotation_text(pos, i)
            if self.task == 'Circle':
                plt.text(2.2, 0, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='left', va='center')

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
            plt.text(5.2, 0, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7), ha='left', va='center')


    def _create_annotation_text(self, pos, i):
        """Create the annotation text for a single frame."""
        state_text = "State Info:"
        x_text = f"x: {pos[0]:.2f}"
        y_text = f"y: {pos[1]:.2f}"
        theta_text = f"\u03B8: {self.thetas[i]*180/np.pi:.2f}"
        v_x_text = fr"$v_y$: {self.velocities[i][0]:.2f}"
        v_y_text = fr"$v_y$: {self.velocities[i][1]:.2f}"
        action_text = "Action Taken:"
        force_text = f"Force: {self.actions[i][0]:.2f}"
        omega_text = f"\u03C9: {self.actions[i][1]:.2f}"

        text = [state_text, x_text, y_text, theta_text, v_x_text, v_y_text, action_text, force_text, omega_text]
        text = '\n'.join(text)
        return text
    
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

    def _log_metrics(self) -> None:
        """Log metrics to Wandb at each step."""
        if len(self.all_rewards) > self.sliding_window:
    
            if self.params['base'].get('wandb_enabled', False):
                wandb.log({
                    "Reward": sum(self.all_rewards[-self.sliding_window:]),
                    "Cost": sum(self.all_costs[-self.sliding_window:]),
                    "No. of Corrective Actions": sum(self.all_corrective_actions[-self.sliding_window:]),
                    "% Safe Actions": sum(self.all_safe_actions[-self.sliding_window:])/self.sliding_window}),
                # }, step = self.metric_log_step)
                # self.metric_log_step += 1

        if self.steps % self.save_freq == 0 or self.steps == self.params['train']['total_num_steps']:
            self.save_data()
            self.save_model()