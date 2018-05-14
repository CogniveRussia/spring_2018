import warnings
from sklearn.cluster import DBSCAN


class CustomMapper(km.KeplerMapper):
    def map(self,
            projected_X,
            inverse_X=None,
            clusterer=DBSCAN(eps=0.5, min_samples=3),
            nr_cubes=None,
            overlap_perc=None,
            coverer=km.Cover(nr_cubes=10, overlap_perc=0.1),
            nerve=km.GraphNerve()):
        """Apply Mapper algorithm on this projection and build a simplicial complex. Returns a dictionary with nodes and links.
        Parameters
        ----------
        projected_X : Numpy Array
            Output from fit_transform
        inverse_X : Numpy Array
            Original data. If `None`, then use `projected_X` for clustering.
        clusterer:
            Scikit-learn API compatible clustering algorithm. Default: DBSCAN
        nr_cubes : Int
            The number of intervals/hypercubes to create. Default = 10. (DeprecationWarning: define Cover explicitly in future versions)
        overlap_perc : Float
            The percentage of overlap "between" the intervals/hypercubes. Default = 0.1. (DeprecationWarning: define Cover explicitly in future versions)
        coverer : kmapper.Cover
            Cover scheme for lens. Instance of kmapper.cover providing methods `define_bins` and `find_entries`.
        nerve : kmapper.Nerve
            Nerve builder implementing `__call__(nodes)` API
        Returns
        =======
        simplicial_complex : dict
            A dictionary with "nodes", "links" and "meta" information.
        Example
        =======
        >>> simplicial_complex = mapper.map(projected_X, inverse_X=None, clusterer=cluster.DBSCAN(eps=0.5,min_samples=3),nr_cubes=10, overlap_perc=0.1)
        >>>print(simplicial_complex["nodes"])
        >>>print(simplicial_complex["links"])
        >>>print(simplicial_complex["meta"])
        """

        start = datetime.now()

        nodes = defaultdict(list)
        meta = defaultdict(list)
        graph = {}

        # If inverse image is not provided, we use the projection as the inverse image (suffer projection loss)
        if inverse_X is None:
            inverse_X = projected_X

        if nr_cubes is not None or overlap_perc is not None:
            # If user supplied nr_cubes or overlap_perc,
            # use old defaults instead of new Cover
            nr_cubes = nr_cubes if nr_cubes else 10
            overlap_perc = overlap_perc if overlap_perc else 0.1
            self.coverer = km.Cover(nr_cubes=nr_cubes,
                                 overlap_perc=overlap_perc)

            warnings.warn(
                "Explicitly passing in nr_cubes and overlap_perc will be deprecated in future releases. Please supply Cover object.", DeprecationWarning)
        else:
            self.coverer = coverer

        if self.verbose > 0:
            print("Mapping on data shaped %s using lens shaped %s\n" %
                  (str(inverse_X.shape), str(projected_X.shape)))

        # Prefix'ing the data with ID's
        ids = np.array([x for x in range(projected_X.shape[0])])
        projected_X = np.c_[ids, projected_X]
        inverse_X = inverse_X

        # Cover scheme defines a list of elements
        bins = self.coverer.define_bins(projected_X)

        # Algo's like K-Means, have a set number of clusters. We need this number
        # to adjust for the minimal number of samples inside an interval before
        # we consider clustering or skipping it.
        cluster_params = clusterer.get_params()
        min_cluster_samples = cluster_params.get("n_clusters", 1)

        if self.verbose > 1:
            print("Minimal points in hypercube before clustering: %d" %
                  (min_cluster_samples))

        # Subdivide the projected data X in intervals/hypercubes with overlap
        if self.verbose > 0:
            bins = list(bins)  # extract list from generator
            total_bins = len(bins)
            print("Creating %s hypercubes." % total_bins)

        for i, cube in enumerate(bins):
            # Slice the hypercube:
            #  gather all entries in this element of the cover
            hypercube = self.coverer.find_entries(projected_X, cube)

            if self.verbose > 1:
                print("There are %s points in cube_%s / %s" %
                      (hypercube.shape[0], i, total_bins))

            # If at least min_cluster_samples samples inside the hypercube
            if hypercube.shape[0] >= min_cluster_samples:

                # Cluster the data point(s) in the cube, skipping the id-column
                # Note that we apply clustering on the inverse image (original data samples) that fall inside the cube.
                inverse_x = inverse_X[[int(nn) for nn in hypercube[:, 0]]]

                clusterer.fit(inverse_x)

                if self.verbose > 1:
                    print("Found %s clusters in cube_%s\n" % (
                        np.unique(clusterer.labels_[clusterer.labels_ > -1]).shape[0], i))

                # TODO: I think this loop could be improved by turning inside out:
                #           - partition points according to each cluster
                # Now for every (sample id in cube, predicted cluster label)
                for a in np.c_[hypercube[:, 0], clusterer.labels_]:
                    if a[1] != -1:  # if not predicted as noise

                        # TODO: allow user supplied label
                        #   - where all those extra values necessary?
                        cluster_id = "cube{}_cluster{}".format(i, int(a[1]))

                        # Append the member id's as integers
                        nodes[cluster_id].append(int(a[0]))
                        meta[cluster_id] = {
                            "size": hypercube.shape[0], "coordinates": cube}
            else:
                if self.verbose > 1:
                    print("Cube_%s is empty.\n" % (i))

        links, simplices = nerve(nodes)

        graph["nodes"] = nodes
        graph["links"] = links
        graph["simplices"] = simplices
        graph["meta_data"] = {
            "projection": self.projection if self.projection else "custom",
            "nr_cubes": self.coverer.nr_cubes,
            "overlap_perc": self.coverer.overlap_perc,
            "clusterer": str(clusterer),
            "scaler": str(self.scaler)
        }
        graph["meta_nodes"] = meta

        # Reporting
        if self.verbose > 0:
            self._summary(graph, str(datetime.now() - start))

        return graph

    def _summary(self, graph, time):
        # TODO: this summary is relevant to the type of Nerve being built.
        links = graph["links"]
        nodes = graph["nodes"]
        nr_links = sum(len(v) for k, v in links.items())

        print("\nCreated %s edges and %s nodes in %s." %
              (nr_links, len(nodes), time))
        
if __name__ == '__main__':
    pass