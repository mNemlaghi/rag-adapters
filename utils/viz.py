from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class ProjectionMaker:
    def __init__(self, X):
        self.projector =  TSNE(n_components=2, learning_rate="auto")
        self.y = self.projector.fit_transform(X)
        
        #Build DF
        self.projections_df = pd.DataFrame(dict(x=self.y[:,0], y = self.y[:,1]))
        self.projections_df['type']= "passage"
        self.projections_df.at[0, "type"] = "question"
        self.projections_df.at[1, "type"] = "relevant"
        self.projections_df.at[2, "type"] = "distractor"


        self.compute_cosine_sims()
        threshold = 0.95
        self.projections_df['close']=self.projections_df['sims']>threshold

    def compute_cosine_sims(self):
        question = self.projections_df.query("type=='question'")[['x','y']].values
        all_points = self.projections_df[['x','y']].values
        sims = np.dot(question, all_points.T) / (np.linalg.norm(question) * np.linalg.norm(all_points, axis=1))
        self.projections_df['sims'] = sims.flatten()
        self.projections_df['close'] = self.projections_df['sims'] > 0.95


    def circle_bounds(self, center_x, center_y, radius):
        xmin = center_x - radius
        xmax = center_x + radius
        ymin = center_y - radius
        ymax = center_y + radius
        return xmin, xmax, ymin, ymax
    

    def plot_projection(self, title: str = "Embedding Projection", point_size: int = 15, 
                        colors: dict = None, show_labels: bool = False, padding: float = 0.1):
        close_points = self.projections_df.query("close==True")
        question = self.projections_df.query("type=='question'")
        
        cx, cy = question['x'].values[0], question['y'].values[0]
        radius = np.sqrt((close_points['x'] - cx) ** 2 + (close_points['y'] - cy) ** 2).max()

        # Calculate the data range
        x_min, x_max = self.projections_df['x'].min(), self.projections_df['x'].max()
        y_min, y_max = self.projections_df['y'].min(), self.projections_df['y'].max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * padding
        y_padding = y_range * padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Ensure the circle fits within the plot
        x_min = min(x_min, cx - radius)
        x_max = max(x_max, cx + radius)
        y_min = min(y_min, cy - radius)
        y_max = max(y_max, cy + radius)

        fig = go.Figure()

        # Add scatter plots for each type
        for type_name in self.projections_df['type'].unique():
            df_type = self.projections_df[self.projections_df['type'] == type_name]
            fig.add_trace(go.Scatter(
                x=df_type['x'], y=df_type['y'],
                mode='markers',
                name=type_name,
                marker=dict(size=point_size, color=colors.get(type_name, None) if colors else None)
            ))

        # Add the circle
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=cx-radius, y0=cy-radius, x1=cx+radius, y1=cy+radius,
            line_color="gold",
            fillcolor="gold",
            opacity=0.2
        )

        # Add labels if requested
        if show_labels:
            for _, row in self.projections_df.iterrows():
                fig.add_annotation(
                    x=row['x'], y=row['y'],
                    text=row['type'],
                    showarrow=False,
                    yshift=10
                )

        # Calculate aspect ratio to maintain circle shape
        aspect_ratio = (x_max - x_min) / (y_max - y_min)

        fig.update_layout(
            title=title,
            template="plotly_dark",
            legend_title_text='Point Type',
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max], scaleanchor="x", scaleratio=1/aspect_ratio),
            width=600,  # You can adjust this
            height=600 / aspect_ratio,  # This ensures the plot is square in data units
        )

        fig.update_layout(xaxis=dict(visible=False),yaxis=dict(visible=False))

        fig.show()

        