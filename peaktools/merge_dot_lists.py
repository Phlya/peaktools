
import click
import pandas as pd
import numpy as np
from cooltools.dotfinder import clust_2D_pixels

from . import cli


# minimal subset of columns to handle:
must_columns = ["chrom1",
                "start1",
                "end1",
                "chrom2",
                "start2",
                "end2",
                "obs.raw",
                "cstart1",
                "cstart2",
                "c_label",
                "c_size",
                "la_exp.donut.value",
                "la_exp.vertical.value",
                "la_exp.horizontal.value",
                "la_exp.lowleft.value"]
                # "la_exp.donut.qval",
                # "la_exp.vertical.qval",
                # "la_exp.horizontal.qval",
                # "la_exp.lowleft.qval"]


def read_validate_dots_list(dots_path):
    # load dots lists ...
    dots = pd.read_table(dots_path)

    try:
        dots_must = dots[must_columns]
    except KeyError as exc_one:
        print("Seems like {} is not in cooltools format or lacks some columns ...".format(dots_path))
        raise exc_one
    # returning the subset:
    return dots_must





@cli.command()
@click.argument(
    "dots_paths",
    metavar="DOTS_PATHS",
    type=click.Path(exists=True, dir_okay=False),
    nargs=-1)
@click.option(
    "--resolutions",
    metavar="RESOLUTIONS",
    help="Allow multiple resolutions, but we are actually expecting 5,"
          "10 and 25 kb resolutions.",
    type=str,
    nargs=1)
# options ...
@click.option(
    '--radius',
    help='Radius for clustering, i.e., to consider'
         'a couple of dots "identical", typically ~20kb.',
    type=int,
    default=20000,
    show_default=True)
@click.option(
    "--HICCUPS_filter", "-f",
    help="Enable HICCUPS-filtering of lowest resolution calls",
    is_flag=True,
    default=False)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--output",
    help="Specify output file name where to store"
         " the results of dot-merger, in a BEDPE-like format.",
    type=str)
@click.option(
    "--bin1_id_name",
    help="Name of the 1st coordinate (row index) to use"
         " for distance calculations and clustering"
         " alternatives include: end1, cstart1 (centroid).",
    type=str,
    default="start1",
    show_default=True)
@click.option(
    "--bin2_id_name",
    help="Name of the 2st coordinate (column index) to use"
         " for distance calculations and clustering"
         " alternatives include: end2, cstart2 (centroid).",
    type=str,
    default="start2",
    show_default=True)



def merge_dot_lists(dots_paths,
                    resolutions,
                    radius,
                    hiccups_filter,
                    verbose,
                    output,
                    bin1_id_name,
                    bin2_id_name):
    resolutions = [int(res) for res in resolutions.split(',')]
    if len(dots_paths) != len(resolutions):
        raise ValueError("Number of dot file paths should match the number of resolutions")
    order = np.argsort(resolutions)
    resolutions = np.asarray(resolutions)[order]
    # load dots lists ...
    dots  = [read_validate_dots_list(dots_path) for dots_path in dots_paths]
    dots = [dots[i] for i in order]

    if verbose:
        # before merging:
        print("Before merging:")
        for i, d in enumerate(dots):
            print("number of dots in {}: {}".format(dots_paths[i], len(d)))
        print("")


    # add some sort of cross-validation later on:

    # add label to each DataFrame:
    for d,r in zip(dots, resolutions):
        d['res'] = r

    # extract a list of chroms:
    chroms = list(dots[0]['chrom1'].drop_duplicates())

    # merge 2 DataFrames and sort (why sorting ?! just in case):
    dots_merged = pd.concat(dots,
                            ignore_index=True).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])

    # l10dat[["chrom1","start1","end1","chrom2","start2","end2"]].copy()

    very_verbose = False
    pixel_clust_list = []
    for chrom in chroms:
        pixel_clust = clust_2D_pixels(dots_merged[(dots_merged['chrom1']==chrom) & \
                                                  (dots_merged['chrom2']==chrom)],
                                      threshold_cluster = radius,
                                      bin1_id_name      = bin1_id_name,
                                      bin2_id_name      = bin2_id_name,
                                      verbose = very_verbose)
        pixel_clust_list.append(pixel_clust)

    # concatenate clustering results ...
    # indexing information persists here ...
    pixel_clust_df = pd.concat(pixel_clust_list, ignore_index=False)
    # now merge pixel_clust_df and dots_merged DataFrame ...
    # # and merge (index-wise) with the main DataFrame:
    dots_merged =  dots_merged.merge(
                                    pixel_clust_df,
                                    how='left',
                                    left_index=True,
                                    right_index=True,
                                    suffixes=('', '_merge'))

    if verbose:
        # report >2 clusters:
        print()
        print("Number of pixels in unwanted >2 clusters: {}".format(len(dots_merged[dots_merged["c_size_merge"]>2])))
        print()


    # next thing we should do is to remove
    # redundant peaks called at 10kb, that were
    # also called at 5kb (5kb being a priority) ...

    # there will be groups (clusters) with > 2 pixels
    # i.e. several 5kb and 10kb pixels combined
    # for now let us keep 5kb, if it's alone or
    # 5kb with the highest obs.raw ...


    # introduce unqie label per merged cluster, just in case:
    dots_merged["c_label_merge"] = dots_merged["chrom1"]+"_"+dots_merged["c_label_merge"].astype(np.str)
    if verbose:
        # final number:
        print("Total number of pixels in input before deduping: {}".format(len(dots_merged)))
        print()
    # now let's just follow HiCCUPs filtering process:
    if hiccups_filter:
        all_peaks = []
        unique_peaks = []
        reproducible_peaks = []
        unique_peaks_around_diag = []
        unique_peaks_strong = []
        for res in resolutions:
            all_res = dots_merged[dots_merged["res"] == res]
            all_peaks.append(all_res)
            unique_peaks.append(all_res[all_res["c_size_merge"] == 1])
            reproducible_peaks.append(all_res[all_res["c_size_merge"]>1 ])
            diagonal_distance = np.abs(unique_peaks[-1][bin1_id_name] - unique_peaks[-1][bin2_id_name])
            unique_peaks_around_diag.append(unique_peaks[-1][diagonal_distance < 220*res]) #Scale with resolution?
            strength_res = unique_peaks[-1]["obs.raw"]
            unique_peaks_strong.append(unique_peaks[-1][strength_res > 100])

            if verbose:
                # describe each category:
                print("number of reproducible_{}_peaks: {}".format(res, len(reproducible_peaks[-1])))
                print("number of unique_{}_peaks: {}".format(res, len(unique_peaks[-1])))
                print("number of unique_{}_peaks_around_diagonal: {}".format(res, len(unique_peaks_around_diag[-1])))
                print("number of unique_{}_peaks_strong: {}".format(res, len(unique_peaks_strong[-1])))
                print()

    # now concatenate these lists ...
        dfs_to_concat = [reproducible_peaks[0],
                         *unique_peaks[1:],
                         unique_peaks_around_diag[0],
                         unique_peaks_strong[0]]

        dots_merged_filtered = pd.concat(dfs_to_concat).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])
        # dedup is required as overlap is unavoidable ...
        dots_merged_filtered = dots_merged_filtered.drop_duplicates(subset=["chrom1",'c_label'])

        if verbose:
            # final number:
            print("number of pixels after the merge: {}".format(len(dots_merged_filtered)))
            print()


    ##############################
    # OUTPUT:
    ##############################
        if output is not None:
            dots_merged_filtered[must_columns].to_csv(
                                            output,
                                            sep='\t',
                                            header=True,
                                            index=False,
                                            compression=None)
        return dots_merged_filtered
    else:
        dots_merged = dots_merged.drop_duplicates(['chrom1', 'c_label'])
        if output is not None:
            dots_merged[must_columns].to_csv(output,
                                             sep='\t',
                                             header=True,
                                             index=False,
                                             compression=None)
    if verbose:
        print("Total number of loops after deduping: {}".format(len(dots_merged)))

    #  return just in case ...
    return dots_merged


#if __name__ == '__main__':
#    merge_dot_lists()



# ###########################################
# # click example ...
# ###########################################
# # import click

# # @click.command()
# # @click.option('--count', default=1, help='Number of greetings.')
# # @click.option('--name', prompt='Your name',
# #               help='The person to greet.')
# # def hello(count, name):
# #     """Simple program that greets NAME for a total of COUNT times."""
# #     for x in range(count):
# #         click.echo('Hello %s!' % name)

# # if __name__ == '__main__':
# #     hello()




















