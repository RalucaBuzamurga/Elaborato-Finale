import numpy as np
import nipy.algorithms.segmentation as nipy_vem
import cv2
import skimage.feature as skfeature
import pandas as pd
import mean_shift as ms
from bipartite_match import bipartite_match
from concurrent.futures import ThreadPoolExecutor, as_completed
import hyperopt as ho
import functools as ft
from math import sqrt
import sqlite3


def f1(tp, fp, fn):
    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))
        return f1


class Optimization:
    def __init__(self, images, true_centers):

        # images = images list
        # true_centers = true centers coordinates list

        self.images = images
        self.true_centers = true_centers

    def segmentation(self, image, mu, sigma, beta, niters=10):

        segmentation = nipy_vem.Segmentation(
            data=image, mu=mu, sigma=sigma, beta=beta)

        segmentation.run(niters=niters)
        map = segmentation.map()
        map = cv2.normalize(
            map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        return map

    def dog_predict(self, image, min_rad, rad_diff, threshold_rel, overlap):

        min_sigma = min_rad / sqrt(3)
        max_sigma = min_sigma + rad_diff / sqrt(3)

        predicted_centers = skfeature.blob_dog(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold_rel * np.max(image),
            overlap=overlap,
        )

        predicted = predicted_centers[:, [2, 1, 0]]
        colnames = ["x", "y", "z"]
        predicted = pd.DataFrame(predicted, columns=colnames)

        return predicted

    def mean_shift_predict(
            self, image, radius, kernel_radius, seeds=None, seeds_img=None
    ):

        mean_shift = ms.SpatialMeanShift(n_dim=3, exclude_border=None)

        if seeds_img is not None:
            seeds = mean_shift.get_seeds(x=seeds_img, radius=radius)
        else:
            seeds = mean_shift.get_seeds(x=image, radius=radius)

        predicted = mean_shift.predict(
            x=map, kernel_radius=kernel_radius, seeds=seeds, n_jobs=1
        )

        predicted = predicted[:, [2, 1, 0]]
        colnames = ["x", "y", "z"]
        predicted = pd.DataFrame(predicted, columns=colnames)

        return predicted

    def evaluate(self, true_centers, predicted_centers, eval_type, max_match_dist=2.5):

        admitted_types = ["complete", "counts", "match", "f1"]
        assert (
            eval_type in admitted_types
        ), f"Wrong evaluation_type provided. {eval_type} not in {admitted_types}."

        if isinstance(true_centers, pd.DataFrame):
            true_centers = pd.DataFrame.to_numpy(true_centers)
        if isinstance(predicted_centers, pd.DataFrame):
            predicted_centers = pd.DataFrame.to_numpy(predicted_centers)

        labeled_centers, matched_centers = bipartite_match(
            true_centers=true_centers,
            pred_centers=predicted_centers,
            max_match_dist=max_match_dist,
        )

        if eval_type == "complete":
            return labeled_centers
        elif eval_type == "match":
            return matched_centers
        else:
            TP = np.sum(labeled_centers.name == "TP")
            FP = np.sum(labeled_centers.name == "FP")
            FN = np.sum(labeled_centers.name == "FN")

            if eval_type == "counts":
                eval_counts = pd.DataFrame([TP, FP, FN]).T
                eval_counts.columns = ["TP", "FP", "FN"]
                return eval_counts
            else:
                return f1(TP, FP, FN)

    def predict_and_evaluate(
            self,
            image,
            true_centers,
            hyper_list,
            mod="dog",
            eval_type="counts",
            max_match_dist=2.5,
            seeds_img=None,
    ):

        map = self.segmentation(
            image=image,
            mu=[hyper_list["mu1"], hyper_list["mu2"]],
            sigma=[hyper_list["sigma1"], hyper_list["sigma2"]],
            beta=hyper_list["beta"],
        )

        admitted_types = ["dog", "mean_shift"]
        assert (
            mod in admitted_types
        ), f"Wrong evaluation_type provided. {mod} not in {admitted_types}."

        if mod == "mean_shift":
            predicted_centers = self.mean_shift_predict(
                image=map,
                kernel_radius=hyper_list["kernel_radius"],
                radius=hyper_list["radius"],
                seeds_img=seeds_img,
            )
        else:
            predicted_centers = self.dog_predict(
                image=map,
                min_rad=hyper_list["min_rad"],
                rad_diff=hyper_list["rad_diff"],
                threshold_rel=hyper_list["threshold_rel"],
                overlap=hyper_list["overlap"],
            )

        return self.evaluate(
            true_centers=true_centers,
            predicted_centers=predicted_centers,
            eval_type=eval_type,
            max_match_dist=max_match_dist,
        )

    def objective(
            self,
            hyper_list,
            mod="dog",
            n_cpu=None,
            max_match_dist=2.5,
            database=None,
            table=None,
    ):

        TP = 0
        FP = 0
        FN = 0

        if database is not None:
            db = database
            conn = sqlite3.connect(db)
            c = conn.cursor()

        with ThreadPoolExecutor(n_cpu) as executor:
            futures = {
                executor.submit(
                    self.predict_and_evaluate,
                    image,
                    true_centers,
                    hyper_list=hyper_list,
                    mod=mod,
                    eval_type="counts",
                    max_match_dist=max_match_dist,
                )
                for image, true_centers in zip(self.images, self.true_centers)
            }
            for future in as_completed(futures):
                res = future.result()

                TP += res["TP"].sum()
                FP += res["FP"].sum()
                FN += res["FN"].sum()

        F1 = f1(TP, FP, FN)

        if database is not None:
            mu1 = hyper_list["mu1"]
            mu2 = hyper_list["mu2"]
            sigma1 = hyper_list["sigma1"]
            sigma2 = hyper_list["sigma2"]
            beta = hyper_list["beta"]

            if mod == "dog":
                min_sigma = hyper_list["min_rad"] / sqrt(3)
                max_sigma = min_sigma + hyper_list["rad_diff"] / sqrt(3)
                threshold_rel = hyper_list["threshold_rel"]
                overlap = hyper_list["overlap"]

                c.execute(
                    "INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)".format(
                        table
                    ),
                    (
                        F1,
                        mu1,
                        mu2,
                        sigma1,
                        sigma2,
                        beta,
                        min_sigma,
                        max_sigma,
                        threshold_rel,
                        overlap,
                    ),
                )

            else:
                radius = hyper_list["radius"]
                kernel_radius = hyper_list["kernel_radius"]
                peaks_dist = hyper_list["peaks_dist"]

                c.execute(
                    "INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)".format(
                        table),
                    (
                        F1,
                        mu1,
                        mu2,
                        sigma1,
                        sigma2,
                        beta,
                        radius,
                        kernel_radius,
                        peaks_dist,
                    ),
                )

            conn.commit()
            conn.close()

        return {"loss": -F1, "status": ho.STATUS_OK}

    def fit(
            self,
            search_space,
            max_match_dist=2.5,
            mod="dog",
            opt_type="rand",
            n_iter=20,
            n_cpu=None,
            database=None,
            table=None,
    ):

        admitted_types = ["rand", "tpe"]
        assert (
            opt_type in admitted_types
        ), f"Wrong evaluation_type provided. {opt_type} not in {admitted_types}."

        obj_wrapper = ft.partial(
            self.objective,
            mod=mod,
            n_cpu=n_cpu,
            max_match_dist=max_match_dist,
            database=database,
            table=table,
        )

        if opt_type == "rand":
            best_par = ho.fmin(
                fn=obj_wrapper,
                space=search_space,
                algo=ho.rand.suggest,
                max_evals=n_iter,
            )
        else:
            best_par = ho.fmin(
                fn=obj_wrapper,
                space=search_space,
                algo=ho.tpe.suggest,
                max_evals=n_iter,
            )

        return best_par
