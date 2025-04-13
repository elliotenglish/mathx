import plotly.graph_objects as go

def tetrahedron_to_triangle(tet):
  tri=[]
  for t in tet:
    tri.append([t[0],t[1],t[2]])
    tri.append([t[1],t[3],t[2]])
    tri.append([t[2],t[3],t[0]])
    tri.append([t[3],t[1],t[0]])
  return tri

def color_str(color):
  return f"rgb({color[0]},{color[1]},{color[2]})"

def generate_mesh3d(vtx,tri,color,flatshading=True):
  return go.Mesh3d(
      x=[v[0] for v in vtx],
      y=[v[1] for v in vtx],
      z=[v[2] for v in vtx],
      # colorbar=dict(title=dict(text='z')),
      # colorscale=[[0, 'gold'],
      #             [0.5, 'mediumturquoise'],
      #             [1, 'magenta']],
      # # Intensity of each vertex, which will be interpolated and color-coded
      # intensity=[0, 0.33, 0.66, 1],
      # i, j and k give the vertices of triangles
      # here we represent the 4 triangles of the tetrahedron surface
      i=[t[0] for t in tri],
      j=[t[1] for t in tri],
      k=[t[2] for t in tri],
      # name='plasma_surface',
      lighting_specular=.8,
      color=color_str(color),
      flatshading=flatshading
      # showscale=True
    )
  
def generate_scatter3d(vtx,color):
  return go.Scatter3d(
    x=[v[0] for v in vtx],
    y=[v[1] for v in vtx],
    z=[v[2] for v in vtx],
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        color=color_str(color),
        opacity=0.8
    )
  )
  
def generate_cone3d(pos,vec):
  return go.Cone(
    x=[p[0] for p in pos],
    y=[p[1] for p in pos],
    z=[p[2] for p in pos],
    u=[v[0] for v in vec],
    v=[v[1] for v in vec],
    w=[v[2] for v in vec],
    sizemode="absolute",
    sizeref=2,
    anchor="tip")

def write_visualization(elements,path):
  fig = go.Figure(data=elements)
  fig.update_scenes(aspectmode="data")
  # fig.update_scenes(aspectratio={"x":1,"y":1,"z":1})
  fig.write_html(path)
