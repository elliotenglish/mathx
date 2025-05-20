import plotly.graph_objects as go

from .tetrahedron import tetrahedron_to_triangle

def add_remove_dupe_simplex(simplex_map,s):
  ss=sorted(s)
  if ss in simplex_map:
    del simplex_map[ss]
  else:
    simplex_map[ss]=s

def get_boundary_faces(tets):
  sorted_map={}
  for tet in tets:
    for tri in tetrahedron_to_triangle(tet):
      tri_s=tuple(sorted(tri))
      if tri_s in sorted_map:
        sorted_map.pop(tri_s)
        # print("popping")
      else:
        sorted_map[tri_s]=tri

  return [v for k,v in sorted_map.items()]

def color_str(color):
  return f"rgb({color[0]},{color[1]},{color[2]})"

def generate_points3d(pts,color):
  return go.Scatter3d(
    x=[p[0] for p in pts],
    y=[p[1] for p in pts],
    z=[p[2] for p in pts],
    marker_color=color_str(color),
    mode="markers",
    marker_size=3
  )
  
def generate_vectors3d(pts,vecs,color):
  return go.Cone(
    x=[p[0] for p in pts],
    y=[p[1] for p in pts],
    z=[p[2] for p in pts],
    u=[v[0] for v in vecs],
    v=[v[1] for v in vecs],
    w=[v[2] for v in vecs],
    colorscale=[[0,color_str(color)]],
    sizemode="raw"
  )
  
def generate_lines3d(lines,color):
  x=[[v[d] for l in lines for v in l+[[None]*3]] for d in range(3)]
  return go.Scatter3d(
    x=x[0],
    y=x[1],
    z=x[2],
    mode="lines+markers",
    marker_size=3,
    marker=dict(color=color_str(color)))

def generate_mesh3d(vtx,tri,color,flatshading=True,wireframe=False):
  if wireframe:
    edges=set()
    for t in tri:
      for i in range(3):
        edges.add((t[i],t[(i+1)%3]))
    x=[]
    y=[]
    z=[]
    for e in edges:
      x.extend([vtx[e[0]][0],vtx[e[1]][0],None])
      y.extend([vtx[e[0]][1],vtx[e[1]][1],None])
      z.extend([vtx[e[0]][2],vtx[e[1]][2],None])
    return go.Scatter3d(
      x=x,
      y=y,
      z=z,
      mode="lines+markers",
      marker_size=3,
      marker=dict(color=color_str(color))
    )
    
  else:
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
        lighting_specular=1.5,
        color=color_str(color),
        flatshading=flatshading,
        # Workaround bug in normals/shading (https://community.plotly.com/t/mesh3d-shading-bug/92247/10)
        lighting=dict(vertexnormalsepsilon=0,
                      facenormalsepsilon=0)
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
