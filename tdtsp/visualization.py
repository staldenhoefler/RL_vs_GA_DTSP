import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import io
from PIL import Image

def get_figure_image(fig):
    """Converts a matplotlib figure to a PIL Image (useful for WandB logging)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def plot_instance_dynamics(instance, sample_edges=5):
    """
    Plots the travel time variation over time bins for a few random edges 
    to visualize the time-dependent properties of the environment.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    n = instance.coords.shape[0]
    t_bins = instance.travel_tensor.shape[0]
    
    # Pick a few random pairs of nodes
    rng = np.random.default_rng(42)
    plotted = 0
    while plotted < sample_edges:
        u = rng.integers(0, n)
        v = rng.integers(0, n)
        if u == v: continue
        
        times = instance.travel_tensor[:, u, v]
        ax.plot(range(t_bins), times, alpha=0.7, label=f"Edge {u}->{v}")
        plotted += 1
        
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Travel Time / Cost")
    ax.set_title("Edge Travel Dynamics Over Horizon")
    # Uncomment to show legend, but usually clutters.
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_tour(instance, tour, title="Tour Visualization"):
    """
    Visualizes the nodes and the taken tour.
    Edges are colored sequentially to indicate the direction and progression of the tour.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    coords = instance.coords
    
    # Plot all nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='lightblue', edgecolor='k', s=60, zorder=5)
    
    # Plot depot distinctively
    ax.scatter(coords[instance.depot, 0], coords[instance.depot, 1], 
               c='red', s=200, marker='*', edgecolor='k', zorder=6, label="Depot")
    
    # Annotate nodes
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x+1, y+1), fontsize=8, zorder=7)
    
    # Draw edges if a tour is provided
    if tour is not None and len(tour) > 1:
        color_map = plt.cm.viridis
        n_edges = len(tour) - 1
        
        for i in range(n_edges):
            u = tour[i]
            v = tour[i+1]
            
            x_vals = [coords[u, 0], coords[v, 0]]
            y_vals = [coords[u, 1], coords[v, 1]]
            
            # Map progression to a color gradient (start=purple, end=yellow)
            color = color_map(i / max(1, n_edges)) 
            
            ax.plot(x_vals, y_vals, color=color, linewidth=2, alpha=0.8, zorder=2)
            
            # Add an arrow at the middle of the segment
            mid_x = (coords[u, 0] + coords[v, 0]) / 2
            mid_y = (coords[u, 1] + coords[v, 1]) / 2
            dx = (coords[v, 0] - coords[u, 0]) * 0.01
            dy = (coords[v, 1] - coords[u, 1]) * 0.01
            
            ax.annotate("", xy=(mid_x + dx, mid_y + dy), xytext=(mid_x, mid_y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=3)

    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def create_edge_weight_gif(instance, save_path="dynamics.gif"):
    """
    Creates an animation of the environment over time, coloring every edge
    by its travel time (weight) to display the time-dependent dependencies.
    """
    coords = instance.coords
    t_bins = instance.travel_tensor.shape[0]
    n = coords.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Pre-compute min and max travel times to set a consistent colormap
    tensor_ma = np.ma.masked_equal(instance.travel_tensor, 0.0)
    min_tt = tensor_ma.min() if t_bins > 0 else 0
    max_tt = tensor_ma.max() if t_bins > 0 else 1
    if min_tt == max_tt:
        max_tt += 1e-5
        
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(vmin=min_tt, vmax=max_tt)
    
    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='lightblue', edgecolor='k', s=60, zorder=5)
    ax.scatter(coords[instance.depot, 0], coords[instance.depot, 1], c='red', s=200, marker='*', edgecolor='k', zorder=6)
    
    ax.set_title("Network Edge Real-time Travel Costs (Bin: 0)")
    ax.set_xticks([])
    ax.set_yticks([])
    
    lines = []
    # Plot all unique undirected edges
    for i in range(n):
        for j in range(i+1, n):
            line, = ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 
                            linewidth=1.5, alpha=0.3, zorder=1)
            lines.append((line, i, j))
            
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Travel Time / Distance (Slower = Darker)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    def update(frame):
        ax.set_title(f"Network Edge Real-time Travel Costs (Time Bin: {frame} / {t_bins-1})")
        for line, i, j in lines:
            # Average weights for undirected plot
            tt = (instance.travel_tensor[frame, i, j] + instance.travel_tensor[frame, j, i]) / 2.0
            color = cmap(norm(tt))
            line.set_color(color)
        return [l[0] for l in lines]
        
    frames = max(1, t_bins)
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=800, blit=False)
    ani.save(save_path, writer='pillow')
    plt.close(fig)
    return save_path

def create_tour_gif(instance, tour, save_path="tour_animation.gif", title="Tour Step-by-Step"):
    """
    Creates an animation of the tour being built step-by-step.
    """
    if tour is None or len(tour) <= 1:
        return None
        
    coords = instance.coords
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Static elements
    ax.scatter(coords[:, 0], coords[:, 1], c='lightblue', edgecolor='k', s=60, zorder=5)
    ax.scatter(coords[instance.depot, 0], coords[instance.depot, 1], 
               c='red', s=200, marker='*', edgecolor='k', zorder=6, label="Depot")
               
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x+1, y+1), fontsize=8, zorder=7)

    ax.grid(True, linestyle='--', alpha=0.5)
    
    color_map = plt.cm.viridis
    n_edges = len(tour) - 1
    
    def init():
        ax.set_title(f"{title} (Start)")
        return []

    def update(frame):
        ax.set_title(f"{title} (Step {frame}/{n_edges})")
        idx = frame - 1
        
        if idx >= n_edges:
            return []
            
        u = tour[idx]
        v = tour[idx+1]
        
        x_vals = [coords[u, 0], coords[v, 0]]
        y_vals = [coords[u, 1], coords[v, 1]]
        
        color = color_map(idx / max(1, n_edges))
        
        line, = ax.plot(x_vals, y_vals, color=color, linewidth=2, alpha=0.8, zorder=2)
        
        mid_x = (coords[u, 0] + coords[v, 0]) / 2
        mid_y = (coords[u, 1] + coords[v, 1]) / 2
        dx = (coords[v, 0] - coords[u, 0]) * 0.01
        dy = (coords[v, 1] - coords[u, 1]) * 0.01
        
        ann = ax.annotate("", xy=(mid_x + dx, mid_y + dy), xytext=(mid_x, mid_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=3)
        return line, ann
        
    ani = animation.FuncAnimation(fig, update, frames=range(1, n_edges + 1), 
                                  init_func=init, interval=600, blit=False, repeat=False)
                                  
    ani.save(save_path, writer='pillow')
    plt.close(fig)
    return save_path

def plot_performance_comparison(ga_cost, rl_cost, ga_time, rl_time):
    """
    Plots a side-by-side comparison of GA and RL results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    algorithms = ['GA', 'RL']
    costs = [ga_cost, rl_cost]
    times = [ga_time, rl_time]
    colors = ['skyblue', 'lightcoral']
    
    # Cost comparison
    bars1 = ax1.bar(algorithms, costs, color=colors)
    ax1.set_title('Total Cost Comparison')
    ax1.set_ylabel('Cost')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')
        
    # Time comparison
    bars2 = ax2.bar(algorithms, times, color=colors)
    ax2.set_title('Inference Time Comparison')
    ax2.set_ylabel('Time (s)')
    ax2.set_yscale('log') # often RL is much faster than GA
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}s', va='bottom', ha='center')
        
    plt.tight_layout()
    return fig
