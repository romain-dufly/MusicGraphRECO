
##-## Cluster analysis ##--##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

##-------------------Data retrieval-------------------

data = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/nodes_complete.csv', sep=';')

embeddings = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/embeddings.csv', sep=';')

data['embedding'] = embeddings['embedding']
data['embedding'] = data['embedding'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=' '))

#pca = PCA(n_components=2)
#pca_result = pca.fit_transform(np.array(data['embedding'].tolist()))
#data['pca-one'] = pca_result[:,0]
#data['pca-two'] = pca_result[:,1]

embeddings2D = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/embeddings2D.csv', sep=';')

data['pca-one'] = embeddings2D['pca-one']
data['pca-two'] = embeddings2D['pca-two']
data['embedding2D'] = data[['pca-one', 'pca-two']].to_numpy().tolist()

print(data.head())

def show_points():
    '''Show the 2D scatter plot of the artists'''
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data['pca-one'], data['pca-two'], c=data['modularity_class'], cmap='Spectral')
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    plt.show()

#show_points()

#-------------------------Recommendation System------------------------------

def convert_preferences_to_data(pref, flat = True):
    '''Converts the preferences to a list of embeddings. If flat, embeddings are 2D, else 16D'''
    d = []
    du = []
    for p in pref:
        artist = data[data['name'] == p[0]]
        if len(artist) > 0:
            for _ in range(p[1]) :
                if flat :
                    d.append(artist['embedding2D'].values[0])
                else :
                    d.append(artist['embedding'].values[0])
            if flat :
                du.append(artist['embedding2D'].values[0])
            else :
                du.append(artist['embedding'].values[0])
        else :
            print('Artist not found : ' + p[0])
    return np.array(d), np.array(du)

def get_recommendations(n=10, id=False):
    '''Returns the n most similar artists to the preferences
    If id, returns the id of the artists, else returns the name'''
    d_liked,du = convert_preferences_to_data(liked, flat=False)
    d_hated, du = convert_preferences_to_data(hated, flat=False)
    print('Data obtained')
    w1 = len(d_liked)/(len(d_liked)+len(d_hated))
    w2 = len(d_hated)/(len(d_liked)+len(d_hated))
    if w2 == 0 :
        d_hated = d_liked
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(d_liked)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=2).fit(d_hated)
    recommendations = []
    for i in data.index:
        if data.iloc[i]['name'] not in [x[0] for x in liked] and data.iloc[i]['name'] not in [x[0] for x in hated]:
            exp1 = np.exp(kde.score_samples(np.array(data.iloc[i]['embedding']).reshape(1, -1)))[0]
            exp2 = np.exp(kde2.score_samples(np.array(data.iloc[i]['embedding']).reshape(1, -1)))[0]
            if id :
                recommendations.append((data.iloc[i]['Id'], exp1*w1 - exp2*w2))
            else :
                recommendations.append((data.iloc[i]['name'], exp1*w1 - exp2*w2))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

def show_kernels():
    '''Show the 2D scatter plot of the artists with the kernel density estimation'''
    global liked, hated
    d_liked, du_liked = convert_preferences_to_data(liked)
    d_hated, du_hated = convert_preferences_to_data(hated)
    w1 = len(d_liked)/(len(d_liked)+len(d_hated))
    w2 = len(d_hated)/(len(d_liked)+len(d_hated))
    if w2 == 0 :
        d_hated = d_liked

    kde = KernelDensity(kernel='gaussian', bandwidth=0.6).fit(d_liked)
    x = np.linspace(-8, 8, 250)
    y = np.linspace(-6, 8, 250)
    X, Y = np.meshgrid(x, y)
    Z = w1 * np.exp(kde.score_samples(np.vstack([X.ravel(), Y.ravel()]).T))

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(d_hated)
    x = np.linspace(-8, 8, 250)
    y = np.linspace(-6, 8, 250)
    X, Y = np.meshgrid(x, y)
    Z -= w2 * np.exp(kde.score_samples(np.vstack([X.ravel(), Y.ravel()]).T))

    Z = Z.reshape(X.shape)
    plt.contourf(X, Y, Z, 20, cmap='Blues')
    plt.colorbar()

    # Add most popular artists from the dataframe
    for i in range(40):
        plt.scatter(data.iloc[i]['embedding2D'][0], data.iloc[i]['embedding2D'][1], c='black')
        plt.text(data.iloc[i]['embedding2D'][0], data.iloc[i]['embedding2D'][1], data.iloc[i]['name'], va='center', ha='center')

    # Show actual preferences points
    plt.scatter([x[0] for x in du_liked], [x[1] for x in du_liked], c='green')
    for i in range(len(du_liked)):
        plt.text(du_liked[i][0], du_liked[i][1], liked[i][0], va='center', ha='center')
    
    if w2 != 0 :
        plt.scatter([x[0] for x in du_hated], [x[1] for x in du_hated], c='red')
        for i in range(len(du_hated)):
            plt.text(du_hated[i][0], du_hated[i][1], hated[i][0], va='center', ha='center')
    
    # Get recommendations and add them to the plot
    names = [x[0] for x in get_recommendations()]
    embeddings = []
    for n in names:
        print(n)
        print(data[data['name'] == n]['embedding2D'])
        embeddings.append(data[data['name'] == n]['embedding2D'].values[0])
    plt.scatter([x[0] for x in embeddings], [x[1] for x in embeddings], c='white')
    for i in range(len(embeddings)):
        plt.text(embeddings[i][0], embeddings[i][1], names[i], va='center', ha='center')

    plt.show()

def recommendation_score(artist):
    '''Returns a score between 0 and 1 for the artist, based on preferences'''
    global liked, hated
    artist = data[data['name'] == artist].iloc[0]
    d_liked,du = convert_preferences_to_data(liked, flat=False)
    d_hated, du = convert_preferences_to_data(hated, flat=False)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.65).fit(d_liked)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.65).fit(d_hated)
    exp1 = np.exp(kde.score_samples(np.array(artist['embedding']).reshape(1, -1)))[0]
    exp2 = np.exp(kde2.score_samples(np.array(artist['embedding']).reshape(1, -1)))[0]
    return exp1*len(d_liked)/(len(d_liked)+len(d_hated)) - exp2*len(d_hated)/(len(d_liked)+len(d_hated))


## ---------------------------------Test the model---------------------------------

liked = [('Taylor Swift', 2), ('Ed Sheeran',1), ('Caravan Palace',2), ('Arctic Monkeys',2), 
               ('Billie Eilish',1), ('Dua Lipa',1), ('Jasmine Thompson',1), 
               ('Saint Motel',1), ('The Weeknd',3), ('Twenty One Pilots',1), ('Martin Garrix',1), 
               ('Lana Del Rey', 1)]
hated = []
#show_kernels()
#print(get_recommendations())

liked = [('Arctic Monkeys',1),]
hated = []
#show_kernels()
#print(get_recommendations())


## Compare the recommendations of the model with the related artists of Spotify

related = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/related.csv', sep=';')

def compare(art_id):
    global liked, hated
    rel_single = related[related['id'] == art_id]
    rel_list = []
    for i in range(len(rel_single['related'].values[0].split(', '))-1):
        if i%2 == 1:
            rel_list.append(str(rel_single['related'].values[0].split(', ')[i][1:-2]))
    rel_list[-1] = rel_list[-1][:-1]
    
    liked = [(data[data['Id'] == art_id]['name'].values[0], 1)]
    hated = []
    recommendations = get_recommendations(id=True)

    count = 0
    for i in range(len(recommendations)):
        if recommendations[i][0] in rel_list:
            count += 1
    print(count)
    return count

def rel_hist(n):
    '''Plots a histogram of the number of Spotify-related artists in the recommendations'''
    values = []
    for i in range(n):
        print(related['name'].values[i])
        values.append(compare(related['id'].values[i]))
    plt.hist(np.array(values), bins=10, align='left', edgecolor='black', linewidth=1.2)
    plt.xlabel('Number of related artists in 10 recommendations')
    plt.xlim = (0,10)
    plt.show()

#rel_hist(250)