from base import Table


def imputation(self, targets, missing=None, method="carry forward", sources=None, tqdm=_tqdm):
    """
    In statistics, imputation is the process of replacing missing data with substituted values.

    See more: https://en.wikipedia.org/wiki/Imputation_(statistics)

    Args:
        table (Table): source table.

        targets (str or list of strings): column names to find and
            replace missing values

        missing (any): value to be replaced

        method (str): method to be used for replacement. Options:

            'carry forward':
                takes the previous value, and carries forward into fields
                where values are missing.
                +: quick. Realistic on time series.
                -: Can produce strange outliers.

            'mean':
                calculates the column mean (exclude `missing`) and copies
                the mean in as replacement.
                +: quick
                -: doesn't work on text. Causes data set to drift towards the mean.

            'mode':
                calculates the column mode (exclude `missing`) and copies
                the mean in as replacement.
                +: quick
                -: most frequent value becomes over-represented in the sample

            'nearest neighbour':
                calculates normalised distance between items in source columns
                selects nearest neighbour and copies value as replacement.
                +: works for any datatype.
                -: computationally intensive (e.g. slow)

        sources (list of strings): NEAREST NEIGHBOUR ONLY
            column names to be used during imputation.
            if None or empty, all columns will be used.

    Returns:
        table: table with replaced values.
    """
    if isinstance(targets, str) and targets not in self.columns:
        targets = [targets]
    if isinstance(targets, list):
        for name in targets:
            if not isinstance(name, str):
                raise TypeError(f"expected str, not {type(name)}")
            if name not in self.columns:
                raise ValueError(f"target item {name} not a column name in self.columns:\n{self.columns}")
    else:
        raise TypeError("Expected source as list of column names")

    if method == "nearest neighbour":
        if sources in (None, []):
            sources = self.columns
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(sources, list):
            for name in sources:
                if not isinstance(name, str):
                    raise TypeError(f"expected str, not {type(name)}")
                if name not in self.columns:
                    raise ValueError(f"source item {name} not a column name in self.columns:\n{self.columns}")
        else:
            raise TypeError("Expected source as list of column names")

    methods = ["nearest neighbour", "mean", "mode", "carry forward"]

    if method == "carry forward":
        new = Table()
        for name in self.columns:
            if name in targets:
                data = self[name][:]  # create copy
                last_value = None
                for ix, v in enumerate(data):
                    if v == missing:  # perform replacement
                        data[ix] = last_value
                    else:  # keep last value.
                        last_value = v
                new[name] = data
            else:
                new[name] = self[name]

        return new

    elif method in {"mean", "mode"}:
        new = Table()
        for name in self.columns:
            if name in targets:
                col = self[name].copy()
                assert isinstance(col, Column)
                stats = col.statistics()
                new_value = stats[method]
                col.replace(target=missing, replacement=new_value)
                new[name] = col
            else:
                new[name] = self[name]  # no entropy, keep as is.

        return new

    elif method == "nearest neighbour":
        new = self.copy()
        norm_index = {}
        normalised_values = Table()
        for name in sources:
            values = self[name].unique().tolist()
            values = sortation.unix_sort(values, reverse=False)
            values = [(v, k) for k, v in values.items()]
            values.sort()
            values = [k for _, k in values]

            n = len([v for v in values if v != missing])
            d = {v: i / n if v != missing else math.inf for i, v in enumerate(values)}
            normalised_values[name] = [d[v] for v in self[name]]
            norm_index[name] = d
            values.clear()

        missing_value_index = self.index(*targets)
        missing_value_index = {
            k: v for k, v in missing_value_index.items() if missing in k
        }  # strip out all that do not have missings.
        ranks = set()
        for k, v in missing_value_index.items():
            ranks.update(set(k))
        item_order = sortation.unix_sort(list(ranks))
        new_order = {tuple(item_order[i] for i in k): k for k in missing_value_index.keys()}

        with tqdm(unit="missing values", total=sum(len(v) for v in missing_value_index.values())) as pbar:
            for _, key in sorted(new_order.items(), reverse=True):  # Fewest None's are at the front of the list.
                for row_id in missing_value_index[key]:
                    err_map = [0.0 for _ in range(len(self))]
                    for n, v in self.to_dict(
                        columns=sources, slice_=slice(row_id, row_id + 1, 1)
                    ).items():  # self.to_dict doesn't go to disk as hence saves an IOP.
                        v = v[0]
                        norm_value = norm_index[n][v]
                        if norm_value != math.inf:
                            err_map = [e1 + abs(norm_value - e2) for e1, e2 in zip(err_map, normalised_values[n])]

                    min_err = min(err_map)
                    ix = err_map.index(min_err)

                    for name in targets:
                        current_value = new[name][row_id]
                        if current_value != missing:  # no need to replace anything.
                            continue
                        if new[name][ix] != missing:  # can confidently impute.
                            new[name][row_id] = new[name][ix]
                        else:  # replacement is required, but ix points to another missing value.
                            # we therefore have to search after the next best match:
                            tmp_err_map = err_map[:]
                            for _ in range(len(err_map)):
                                tmp_min_err = min(tmp_err_map)
                                tmp_ix = tmp_err_map.index(tmp_min_err)
                                if row_id == tmp_ix:
                                    tmp_err_map[tmp_ix] = math.inf
                                    continue
                                elif new[name][tmp_ix] == missing:
                                    tmp_err_map[tmp_ix] = math.inf
                                    continue
                                else:
                                    new[name][row_id] = new[name][tmp_ix]
                                    break

                    pbar.update(1)
        return new

    else:
        raise ValueError(f"method {method} not recognised amonst known methods: {list(methods)})")