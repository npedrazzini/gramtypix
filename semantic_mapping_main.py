import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
import skbio
from pykrige import OrdinaryKriging

# Define variables
with open("./src/config.yaml", "r") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# --- Relevant vars
## -- General
mapsoutputdir = configs['plotting']['mapsoutputdir']
krigingstatsdir = configs['plotting']['krigingstatsdir']
paralleldatapath = configs['plotting']['paralleldatapath']
glottologmappingpath = configs['plotting']['glottologmappingpath']
languages = configs['plotting']['languages']

# Import Glottolog mapping from iso codes to language names and families for mapping
lang_names_fams_df = pd.read_csv(glottologmappingpath)

# This assumes the first two columns are the sentence and the sentence_id, hence why they are removed
# Change or remove df = df.iloc[:, 2:] accordingly
parallel_df = pd.read_csv(paralleldatapath, low_memory=False)
parallel_df = parallel_df.iloc[:, 3:]

# Convert strings to numeric using LabelEncoder
label_encoder = LabelEncoder()
df_encoded = parallel_df.apply(label_encoder.fit_transform)

# Compute pairwise Hamming distances
distances = pdist(df_encoded.values, metric='hamming')

# Convert to square matrix
dist_matrix = squareform(distances)

# Save the matrix to a CSV file without header if needed, else comment out
# pd.DataFrame(dist_matrix).to_csv('matrix.csv', index=False, header=False)

# Sanity check
print(dist_matrix.shape)

## Metric Multi Dimensional Scaling (or 'Principal Coordinate Analysis')
PCoA = skbio.stats.ordination.pcoa(dist_matrix)
dims1_2df = PCoA.samples[['PC1', 'PC2']]
plt.scatter(dims1_2df['PC1'],dims1_2df['PC2'],c='black')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Unlabelled MDS map')
plt.axis('square')
plt.savefig(f'{mapsoutputdir}plainmds.png')
plt.show()

## Calculate boundaries for better areas (similarly to `boundary' in qlcVisualize in R)
testdf = dims1_2df[['PC1', 'PC2']]

grid=10
tightness="auto"

p = np.array(testdf)
if tightness == "auto":
    # This is the Python translation of the R function MASS::bandwidth.nrd, works exactly
    r = np.percentile(p[:,0], [25, 75])
    h = (r[1] - r[0]) / 1.34
    tightness = 4 * 1.06 * min(np.sqrt(np.var(p[:,0])), h) * len(p[:,0])**(-1/5)

h = tightness

x = p[:,0]
y = p[:,1]
lims = [np.min(x), np.max(x), np.min(y), np.max(y)]
h = tightness
grid = 10
n = grid
nx = len(x)
print('nx',nx)
n = np.repeat(n, 2)
print('n',n)
gx = np.linspace(lims[0], lims[1], n[0])
print('gx',gx)
gy = np.linspace(lims[2], lims[3], n[1])
print('gy',gy)
h = np.repeat(h, 2)
print('h',h)
h = h / 4
print('h',h)
ax = np.subtract.outer(gx, x) / h[0]
print('ax',ax)
ay = np.subtract.outer(gy, y) / h[1]
print('ay',ay)
matrax = np.reshape(norm.pdf(ax), (-1, nx), order='F')
matray = np.reshape(norm.pdf(ay), (-1, nx), order='F')
z = np.dot(matrax, matray.T) / (nx * h[0] * h[1])
k = {'x': gx, 'y': gy, 'z': z}

density=0.02
grid=10
box_offset=0.1
tightness=tightness
manual=None
plot=True
zeros = np.where(k['z'] < density)
print('zeros',zeros)
zeroX = k['x'][zeros[1]]
zeroY = k['y'][zeros[0]]
rX = np.ptp(x) * box_offset
rY = np.ptp(y) * box_offset

def bandwidth_nrd(x):
    r = np.percentile(x, [25, 75])
    h = (r[1] - r[0]) / 1.34
    return 4 * 1.06 * min(np.sqrt(np.var(x)), h) * len(x) ** (-1/5)

pXmin = np.min(k['x']) - rX
pXmax = np.max(k['x']) + rX
pYmin = np.min(k['y']) - rY
pYmax = np.max(k['y']) + rY

borderX = np.concatenate([
    k['x'],                  # k$x
    k['x'],                  # k$x
    np.tile(pXmin, len(k['y'])),  # rep(pXmin, times = length(k$y))
    np.tile(pXmax, len(k['y'])),  # rep(pXmax, times = length(k$y))
    np.array([pXmin]), np.array([pXmin]), np.array([pXmax]), np.array([pXmax])   # pXmin, pXmin, pXmax, pXmax
])

borderY = np.concatenate([
    np.tile(pYmin, len(k['x'])),  # rep(pYmin, length(k$x))
    np.tile(pYmax, len(k['x'])),  # rep(pYmax, length(k$x))
    k['y'],                       # k$y
    k['y'],                       # k$y
    np.array([pYmin]), np.array([pYmax]), np.array([pYmin]), np.array([pYmax])
])

zeros = np.column_stack((np.concatenate([zeroX, borderX, [None]]), np.concatenate([zeroY, borderY, [None]])))
mask = np.all(zeros != None, axis=1)

# Apply the mask to filter out rows with None values
arr_filtered = zeros[mask]
h0 = np.zeros(arr_filtered.shape[0], dtype=int)
x = np.vstack([np.array(dims1_2df[['PC1', 'PC2']]), arr_filtered])

x1 = np.linspace(min(x[:,0])-0.07,max(x[:,0])+0.07,100)
y1 = np.linspace(min(x[:,1])-0.07,max(x[:,1])+0.07,100)
xgrid,ygrid = np.meshgrid(x1,y1)

# Assuming you want to process the top N items
num_top_items = 5  # Adjust this as needed

# Other variables and configurations
levels = [0.85, 0.90, 0.95]
freq_threshold = round(len(parallel_df) / 100 * 10) - 10
markers = [".", ",", "o", "v", "^", "<", ">"]
colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']

for language in languages:

    print(f'Now processing {language}')

    langname = list(lang_names_fams_df[lang_names_fams_df['iso']==language]['name'])[0]
    nameoffam = list(lang_names_fams_df[lang_names_fams_df['iso']==language]['nameoffam'])[0]

    occurrences = parallel_df.filter(like=language).copy()
    language_code = occurrences.columns[0]

    top_items = list(occurrences[language_code].value_counts().head(num_top_items).index)

    other_label_added = False

    for item_index, top_item in enumerate(top_items):
        occurrences[top_item] = occurrences[language_code] == top_item

        occurrences['x'] = dims1_2df['PC1'].values
        occurrences['y'] = dims1_2df['PC2'].values

        Y = np.concatenate((occurrences[top_item], h0))

        # Perform Ordinary Kriging
        OK = OrdinaryKriging(x[:, 0], x[:, 1], Y,
                             variogram_model='gaussian',
                             nlags=4,
                             coordinates_type='geographic',
                             variogram_parameters={'sill': 0.7, 'range': 0.4, 'nugget': 0.2})

        zgrid, _ = OK.execute('grid', x1, y1, mask=False)
        zgrid_normalized = (zgrid - zgrid.min()) / (zgrid.max() - zgrid.min())

        # Plotting
        means = occurrences[occurrences[top_item] == 1]
        color = colors[item_index % len(colors)] if top_item != 'NOMATCH' else 'black'
        marker = markers[item_index % len(markers)] if top_item != 'NOMATCH' else '>'

        if len(means) >= freq_threshold:
            plt.scatter(means['x'], means['y'], c=color, s=7, marker=marker, label=f'{top_item}', alpha=0.5)

            plot = plt.contour(xgrid, ygrid, zgrid_normalized, levels=levels, colors=color)
            plt.clabel(plot, inline=True, fontsize=7)
        else:
            plt.scatter(means['x'], means['y'], c='gray', s=5, marker='.', alpha=0.4)
            if not other_label_added and top_item != 'NOMATCH':
                plt.scatter([], [], c='gray', s=7, marker='>', label='other', alpha=0.4)
                other_label_added = True

        if occurrences[top_item].sum() >= freq_threshold:
            plot = plt.contour(xgrid, ygrid, zgrid_normalized, levels=levels, colors=color)
            plt.clabel(plot, inline=True, fontsize=7)

            # Check if test points are inside any of the contour paths
            isitinarea = []
            for (point1, point2) in np.array(testdf[['PC1', 'PC2']]):
                test_point = (point1, point2)
                result = any(path.contains_point(test_point) for path in plot.collections[0].get_paths() if plot.levels[0] == 0.85)
                isitinarea.append(1 if result else 0)

            isitinarea = np.array(isitinarea)
            np.save(f'{krigingstatsdir}{language_code}_{top_item}.npy', isitinarea)

    # Rest of the code remains unchanged

    plt.axis('square')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'{langname} [{language}] ({nameoffam})')
    plt.legend(fontsize="9")
    plt.savefig(f'{mapsoutputdir}{language}-kriging.png', dpi=300)
    plt.close()