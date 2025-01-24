
import networkx as nx
import numpy as np

import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from scipy.integrate import quad

import graph

def display_graph(X, Z_vrai):
  
  #on crée une liste parti où parti[i] correspond au numéro de parti du noeud i
  parti = [list(Z_vrai[i]).index(1)+1 for i in range(len(Z_vrai))]
  parti = np.array(parti).T
  
  #on crée le graphe 
  G = nx.from_numpy_array(X)
  pos = nx.spring_layout(G, k=0.2)
  x_nodes = [pos[node][0] for node in G.nodes()]
  y_nodes = [pos[node][1] for node in G.nodes()]
  node_labels = [str(node) for node in G.nodes()]
      
  
  # On associe une couleur à chaque noeud en focntion de son parti
  block_colors = {1: 'green', 2: 'darkblue', 3: 'lightblue', 4: 'red', 5: 'pink', 6: 'darkred', 7: 'black', 8: 'yellow', 9: 'grey', 10: 'cyan'}  # Choisir les couleurs pour chaque bloc
  # Assigner les couleurs aux nœuds
  node_colors = [block_colors[parti[i]] for i in range(len(parti))]  
  block_colors = {i:block_colors[i] for i in block_colors if i<=len(Z_vrai)}
  
       
  legend_labels = [
          ("Les Verts", 'green'),
          ("UMP", 'darkblue'),
          ("UDF", 'lightblue'),
          ("PS", 'red'),
          ("Parti Radical de Gauche", 'pink'),
          ("PCF - LCR", 'darkred'),
          ("FN - MNR - MPF", 'black'),
          ("Libéraux", 'yellow'),
          ("Commentateurs Analystes", 'grey'),
          ("Cap21", 'cyan')
      ]
          
  legend = [
              go.layout.Annotation(
                  x=0.95, y=1 - (i * 0.05),  # Positionnement vertical
                  xref="paper", yref="paper",  # Coordonnées relatives au graphe
                  text=f'{label}',
                  showarrow=False,
                  font=dict(size=12, color=color),
                  align="left"
              )
          for i, (label, color) in enumerate(legend_labels)
      ]
        
  # Visualiser le graphe avec NetworkX et Matplotlib
  edges_x = []
  edges_y = []
  for edge in G.edges():
      x0, y0 = pos[edge[0]]
      x1, y1 = pos[edge[1]]
      edges_x.append(x0)
      edges_x.append(x1)
      edges_y.append(y0)
      edges_y.append(y1)
  
  
    # Créer un objet plotly pour les nœuds
  node_trace = go.Scatter(
      x=x_nodes,
      y=y_nodes,
      mode='markers+text',
      hoverinfo='text',
      marker=dict(
          showscale=True,
          colorscale='YlGnBu',  
          size=10,
          color=node_colors,  # Affecter les couleurs aux nœuds
          line_width=2
      )
  )
  
  # Créer un objet plotly pour les arêtes
  edge_trace = go.Scatter(
      x=edges_x,
      y=edges_y,
      line=dict(width=0.5, color='gray'),
      hoverinfo='none',
      mode='lines'
  )
  
  
  
  # Créer le graphique
  fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      showlegend=False,
                      hovermode='closest',
                      title="Graphe de la blogosphère française ",
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=False, zeroline=False),
                      plot_bgcolor='white',
                      margin=dict(l=0, r=0, b=0, t=0),
                      annotations = legend
      
                  ))
  
          
  
  #Afficher le graphe
  fig.show()
  


def display_graphon(mu,pi):
    #j'ai crée des copies des tableaux comme ça quand je faisais mes tests je modofiais pas les vrai mu et pi à chaque fois
    mu_2 = np.array([[mu[i][j] for j in range(n)] for i in range(n)])
    pi_2 = np.array([pi[i] for i in range(n)])

    block_colors = {1: 'green', 2: 'blue', 3: 'lightblue', 4: 'red', 5: 'pink', 6: 'darkred', 7: 'brown', 8: 'yellow', 9: 'grey', 10: 'cyan'}  # Choisir les couleurs pour chaque bloc

    sorted = np.argsort(mu_2@pi_2)  # Assurer une somme croissante pour les intégrales

    pi_2 = [pi_2[i] for i in sorted]

    # Calcul des bornes cumulées
    cumu_pi = np.cumsum(pi_2)
    cumu_pi = np.insert(cumu_pi, 0, 0)  # Ajouter 0 au début

    # Définir les paramètres de la grille
    u = np.linspace(0, 1, 100)
    v = np.linspace(0, 1, 100)
    u, v = np.meshgrid(u, v)
    face_colors = np.zeros_like(u, dtype=object)

    z = np.zeros_like(u)
    for l1 in range(1, n + 1):
        for l2 in range(1, n + 1):
            mask = (
                (u >= cumu_pi[l1 - 1]) & (u < cumu_pi[l1]) &
                (v >= cumu_pi[l2 - 1]) & (v < cumu_pi[l2])
            )
            z[mask] = mu_2[sorted[l1 - 1]][sorted[ l2 - 1]]
            mask = (
                (u >= cumu_pi[l1 - 1]) & (u < cumu_pi[l1]) &
                (v >= cumu_pi[l2 - 1]-0.02) & (v < cumu_pi[l2])
            )
            if l1 == l2:
                face_colors[mask] = block_colors[sorted[l1-1]+1]
            else:
                face_colors[mask] = "white"


    # Créer une figure et une projection 3D
    fig = plt.pyplot.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    # Appliquer les couleurs personnalisées
    cmap = ListedColormap([block_colors[i] for i in range(1, len(block_colors) + 1)])
    surf = ax.plot_surface(u, v, z, facecolors=face_colors, edgecolor='none')
    patches = [
        mpatches.Patch(color=block_colors[i], label=f"{parti_politique[i - 1]}")
        for i in range(1, len(parti_politique) + 1)
    ]

    plt.pyplot.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), title="Partis politiques")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    plt.pyplot.show()


def display_graphon_integral(mu,pi):
    def integrand(v, u, pi, mu):
        return graph.Graph.graphon_croiss(pi, mu, u, v)[0]
    #on cacule l'intégrande avec un pas de 0,002 car la fonction W est très sensible pour les petites valeurs
    rez = []
    for u in np.linspace(0, 1, 500):
        integral_result, error =quad(integrand, 0, 1, args=(u, pi, mu))
        rez.append(integral_result)
    plt.plot(np.linspace(0, 1, 500), rez, label="$\int_0^1 W_{croissante}(u, v) \, dv$")
    plt.title("Graphe de $\int_0^1 W_{croissante}(u, v) \, dv$ pour $u \in [0,1]$")
    plt.xlabel('u')
    plt.legend()
    plt.show()
    return rez
