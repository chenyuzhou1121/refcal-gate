"""Rank normal reference images and construct six query/reference control inputs."""

import hashlib


CONTROLS = ("query_only", "correct_reference", "same_category_distractor",
            "blurred_reference", "blank_reference", "wrong_category")


class ReferenceControls:
    """Cache whole-image embeddings from category -> reference path lists.

    The encoder must expose encode_images, returning unit-normalized features.
    Each category needs at least two references; blur_radius must be nonnegative.
    """

    def __init__(self, encoder, references, blur_radius=12.0):
        from PIL import Image

        if blur_radius < 0:
            raise ValueError("Blur radius must be nonnegative")
        self.encoder = encoder
        self.references = references
        self.blur_radius = blur_radius
        self.embeddings = {}
        for category, paths in references.items():
            if len(paths) < 2:
                raise ValueError("Reference controls require at least two references per category")
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            self.embeddings[category] = encoder.encode_images(images)

    def rank(self, image, category):
        """Return (path, cosine similarity) pairs in descending order for a PIL query."""
        if category not in self.references:
            raise ValueError(f"No references for category: {category}")
        query = self.encoder.encode_images([image])[0]
        similarities = self.embeddings[category] @ query
        order = sorted(range(len(similarities)), key=lambda index: -float(similarities[index]))
        return [(self.references[category][index], float(similarities[index])) for index in order]

    def select(self, sample, ranked, control):
        """Return (PIL reference, source path), or (None, None) for query_only.

        correct_reference uses the top-ranked image; same_category_distractor
        uses the last-ranked image. Blur and blank controls modify the top image.
        wrong_category deterministically chooses a path outside sample.category.
        ranked is the output of rank for this query and category.
        """
        from PIL import Image, ImageFilter

        if control not in CONTROLS:
            raise ValueError(f"Unknown reference control: {control}")
        if control == "query_only":
            return None, None
        if control == "wrong_category":
            candidates = [path for category in sorted(self.references) if category != sample.category
                          for path in sorted(self.references[category])]
            if not candidates:
                raise ValueError("Wrong-category controls require at least two categories")
            # Stable hashing keeps the choice independent of Python's hash seed.
            digest = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
            path = candidates[int(digest[:12], 16) % len(candidates)]
        else:
            path = ranked[-1 if control == "same_category_distractor" else 0][0]
        with Image.open(path) as source:
            reference = source.convert("RGB")
        if control == "blurred_reference":
            reference = reference.filter(ImageFilter.GaussianBlur(self.blur_radius))
        elif control == "blank_reference":
            reference = Image.new("RGB", reference.size, (128, 128, 128))
        return reference, path
