import os

import numpy as np
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - depends on environment
    linear_sum_assignment = None

from clashroyalebuildabot.constants import CARD_CONFIG
from clashroyalebuildabot.constants import IMAGES_DIR
from clashroyalebuildabot.namespaces.cards import Cards
from error_handling import WikifiedError


class CardDetector:
    HAND_SIZE = 5
    MULTI_HASH_SCALE = 0.355
    MULTI_HASH_INTERCEPT = 163

    def __init__(self, cards, hash_size=8, grey_std_threshold=5):
        self.cards = cards
        self.hash_size = hash_size
        self.grey_std_threshold = grey_std_threshold

        self.cards.extend([Cards.BLANK for _ in range(5)])
        self.card_hashes = self._calculate_card_hashes()

    def _calculate_multi_hash(self, image):
        gray_image = self._calculate_hash(image)
        light_image = (
            self.MULTI_HASH_SCALE * gray_image + self.MULTI_HASH_INTERCEPT
        )
        dark_image = (
            gray_image - self.MULTI_HASH_INTERCEPT
        ) / self.MULTI_HASH_SCALE
        multi_hash = np.vstack([gray_image, light_image, dark_image]).astype(
            np.float32
        )
        return multi_hash

    def _calculate_hash(self, image):
        return np.array(
            image.resize(
                (self.hash_size, self.hash_size), Image.Resampling.BILINEAR
            ).convert("L"),
            dtype=np.float32,
        ).ravel()

    def _calculate_card_hashes(self):
        card_hashes = np.zeros(
            (
                len(self.cards),
                3,
                self.hash_size * self.hash_size,
                self.HAND_SIZE,
            ),
            dtype=np.float32,
        )
        try:
            for i, card in enumerate(self.cards):
                path = os.path.join(IMAGES_DIR, "cards", f"{card.name}.jpg")
                pil_image = Image.open(path)

                multi_hash = self._calculate_multi_hash(pil_image)
                card_hashes[i] = np.tile(
                    np.expand_dims(multi_hash, axis=2), (1, 1, self.HAND_SIZE)
                )
        except Exception as e:
            raise WikifiedError(
                "005", "Can't load cards and their images."
            ) from e
        return card_hashes

    def _detect_cards(self, image):
        crops = [image.crop(position) for position in CARD_CONFIG]
        crop_hashes = np.array(
            [self._calculate_hash(crop) for crop in crops]
        ).T
        hash_diffs = np.mean(
            np.amin(np.abs(crop_hashes - self.card_hashes), axis=1), axis=1
        ).T
        _, idx = self._linear_sum_assignment(hash_diffs)
        cards = [self.cards[i] for i in idx]

        return cards, crops

    @staticmethod
    def _linear_sum_assignment(cost_matrix: np.ndarray):
        """
        Assign each row to a unique column with minimum cost.

        SciPy provides `linear_sum_assignment`, but on Android/Termux SciPy is
        often unavailable. Because our hand size is tiny (5), a brute-force
        fallback is fast enough.
        """

        if linear_sum_assignment is not None:
            return linear_sum_assignment(cost_matrix)

        # Fallback: brute force permutations (HAND_SIZE = 5, n_cards ~= 13)
        import itertools

        n_rows, n_cols = cost_matrix.shape
        if n_cols < n_rows:
            raise ValueError(
                f"cost_matrix must have at least as many columns as rows "
                f"({n_rows=} {n_cols=})"
            )

        best_cost = float("inf")
        best_cols = None
        col_range = range(n_cols)
        # 13P5 = 154,440 worst-case; OK for assist mode.
        for cols in itertools.permutations(col_range, n_rows):
            cost = 0.0
            for r, c in enumerate(cols):
                cost += float(cost_matrix[r, c])
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost = cost
                best_cols = cols

        row_ind = np.arange(n_rows, dtype=np.int64)
        col_ind = np.array(best_cols, dtype=np.int64)
        return row_ind, col_ind

    def _detect_if_ready(self, crops):
        ready = []
        for i, crop in enumerate(crops[1:]):
            std = np.mean(np.std(np.array(crop), axis=2))
            if std > self.grey_std_threshold:
                ready.append(i)
        return ready

    def run(self, image):
        cards, crops = self._detect_cards(image)
        ready = self._detect_if_ready(crops)
        return cards, ready
