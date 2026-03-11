#!/usr/bin/env python3
"""Cuckoo filter — space-efficient probabilistic set membership with deletion support.

One file. Zero deps. Does one thing well.

Unlike Bloom filters, cuckoo filters support deletion and have better lookup
performance. Based on "Cuckoo Filter: Practically Better Than Bloom" (Fan et al., 2014).
"""
import hashlib, struct, sys, random

class CuckooFilter:
    """Cuckoo filter with configurable bucket size and fingerprint bits."""
    MAX_KICKS = 500

    def __init__(self, capacity=1024, bucket_size=4, fp_bits=8):
        self.bucket_size = bucket_size
        self.fp_bits = fp_bits
        self.fp_mask = (1 << fp_bits) - 1
        # Round capacity to power of 2
        self.num_buckets = 1
        while self.num_buckets < capacity // bucket_size:
            self.num_buckets <<= 1
        self.buckets = [[0] * bucket_size for _ in range(self.num_buckets)]
        self.count = 0

    def _hash(self, data):
        if isinstance(data, str):
            data = data.encode()
        return int(hashlib.sha256(data).hexdigest(), 16)

    def _fingerprint(self, item):
        h = self._hash(item)
        fp = h & self.fp_mask
        return fp if fp != 0 else 1  # 0 means empty

    def _index(self, item):
        h = self._hash(item)
        return (h >> self.fp_bits) % self.num_buckets

    def _alt_index(self, i, fp):
        h = self._hash(struct.pack('>Q', fp))
        return (i ^ (h % self.num_buckets)) % self.num_buckets

    def insert(self, item):
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            for j in range(self.bucket_size):
                if self.buckets[i][j] == 0:
                    self.buckets[i][j] = fp
                    self.count += 1
                    return True
        # Must relocate
        i = random.choice([i1, i2])
        for _ in range(self.MAX_KICKS):
            j = random.randrange(self.bucket_size)
            fp, self.buckets[i][j] = self.buckets[i][j], fp
            i = self._alt_index(i, fp)
            for j in range(self.bucket_size):
                if self.buckets[i][j] == 0:
                    self.buckets[i][j] = fp
                    self.count += 1
                    return True
        return False  # Filter is full

    def lookup(self, item):
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item):
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            for j in range(self.bucket_size):
                if self.buckets[i][j] == fp:
                    self.buckets[i][j] = 0
                    self.count -= 1
                    return True
        return False

    def __len__(self):
        return self.count

    def __contains__(self, item):
        return self.lookup(item)

    def load_factor(self):
        total = self.num_buckets * self.bucket_size
        return self.count / total if total else 0.0


def demo():
    cf = CuckooFilter(capacity=256, bucket_size=4, fp_bits=12)
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    for w in words:
        cf.insert(w)
    print(f"Inserted {len(cf)} items (load: {cf.load_factor():.1%})")
    for w in words:
        assert cf.lookup(w), f"{w} should be found"
    print("All items found ✓")
    # False positive check
    fp = sum(1 for i in range(1000) if cf.lookup(f"nonexistent_{i}"))
    print(f"False positives: {fp}/1000 ({fp/10:.1f}%)")
    # Deletion
    cf.delete("banana")
    assert not cf.lookup("banana"), "banana should be deleted"
    print("Deletion works ✓")
    print(f"Final count: {len(cf)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        demo()
    else:
        print(__doc__.strip())
        print("\nUsage: python cuckoo_filter.py test")
