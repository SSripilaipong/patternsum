class ScoreManager:
    def __init__(self):
        self.scores = {}

    def query(self, patterns):
        patterns_mis = []
        patterns_hit = []

        for pattern in patterns:
            scores = self.scores.get(pattern, None)
            if scores:
                pattern._tightness = scores['tightness']
                pattern.fitness = scores['fitness']
                pattern.accuracy = scores['accuracy']
                pattern.match_indexes = scores['match_indexes']
                pattern.match_count = scores['match_count']

                patterns_hit.append(pattern)
            else:
                patterns_mis.append(pattern)

        return patterns_hit, patterns_mis

    def update(self, patterns):
        for pattern in patterns:
            if pattern in self.scores:
                continue

            scores = {
                'tightness': pattern.tightness,
                'fitness': pattern.fitness,
                'accuracy': pattern.accuracy,
                'match_indexes': pattern.match_indexes,
                'match_count': pattern.match_count,
            }
            self.scores[pattern.copy()] = scores
