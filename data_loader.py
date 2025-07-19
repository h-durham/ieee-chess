from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import utils
import torch
import random


THEME_SET = [
    "NONE",
    "advancedPawn",
    "advantage",
    "anastasiaMate",
    "arabianMate",
    "attackingF2F7",
    "attraction",
    "backRankMate",
    "bishopEndgame",
    "bodenMate",
    "capturingDefender",
    "castling",
    "clearance",
    "crushing",
    "defensiveMove",
    "deflection",
    "discoveredAttack",
    "doubleBishopMate",
    "doubleCheck",
    "dovetailMate",
    "enPassant",
    "endgame",
    "equality",
    "exposedKing",
    "fork",
    "hangingPiece",
    "hookMate",
    "interference",
    "intermezzo",
    "kingsideAttack",
    "knightEndgame",
    "long",
    "master",
    "masterVsMaster",
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    "middlegame",
    "oneMove",
    "opening",
    "pawnEndgame",
    "pin",
    "promotion",
    "queenEndgame",
    "queenRookEndgame",
    "queensideAttack",
    "quietMove",
    "rookEndgame",
    "sacrifice",
    "short",
    "skewer",
    "smotheredMate",
    "superGM",
    "trappedPiece",
    "underPromotion",
    "veryLong",
    "xRayAttack",
    "zugzwang",
]

THEME2IDX = {t: idx for (idx, t) in enumerate(THEME_SET)}
IDX2THEME = {idx: t for (t, idx) in enumerate(THEME_SET)}

BUCKETS = [
    (0, 600),
    (600, 900),
    (900, 1200),
    (1200, 1500),
    (1500, 1800),
    (1800, 2100),
    (2100, 2400),
    (2700, 3100),
]
BUCKET_MEANS = [
    400.0,
    750.0,
    1050.0,
    1350.0,
    1650.0,
    1950.0,
    2250.0,
    3000.0,
]

MOVE_BUCKETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def get_bucket_idx(rating):
    for idx, (_, b) in enumerate(BUCKETS):
        if rating < b:
            return idx
    return len(BUCKETS) - 1


def get_move_bucket(moves):
    if moves > 15:
        return len(MOVE_BUCKETS) - 1
    return moves


class SimpleBaselineDataset(Dataset):
    def __init__(self, puzzles_file, filter=False, regularize=False, flip_boards=0.0):
        self.puzzles = pd.read_csv(puzzles_file)
        if filter:
            self.puzzles = self.puzzles.loc[
                (self.puzzles["RatingDeviation"] <= 100.0)
                & (self.puzzles["NbPlays"] >= 500)
                | (self.puzzles["Rating"] <= 1000)
                | (self.puzzles["Rating"] >= 2000)
            ]
        self._regularize = regularize
        self.puzzles["Themes"] = self.puzzles["Themes"].fillna("NONE")
        self.flip_boards = flip_boards

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        fen = self.puzzles.iloc[idx, 1]
        if self._regularize:
            rating = self.puzzles.iloc[idx, 3] + (random.random() * 2 - 1) * 20
        else:
            rating = self.puzzles.iloc[idx, 3]
        moves = self.puzzles.iloc[idx, 2]
        moves = moves.split()

        if random.random() < self.flip_boards:
            fen, moves = utils.flip_board(fen, moves)

        board = utils.ChessBoard(fen)
        board.move(moves[0])
        first_board = board.t
        board.move(moves[1])
        second_board = board.t

        if len(moves) >= 3:
            board.move(moves[2])
            third_board = board.t
        else:
            third_board = torch.zeros_like(first_board)
        if len(moves) >= 4:
            board.move(moves[3])
            fourth_board = board.t
        else:
            fourth_board = torch.zeros_like(first_board)
        if len(moves) >= 5:
            board.move(moves[4])
            fifth_board = board.t
        else:
            fifth_board = torch.zeros_like(first_board)
        if len(moves) >= 6:
            board.move(moves[5])
            sixth_board = board.t
        else:
            sixth_board = torch.zeros_like(first_board)
        if len(moves) >= 7:
            board.move(moves[6])
            seventh_board = board.t
        else:
            seventh_board = torch.zeros_like(first_board)
        if len(moves) >= 8:
            board.move(moves[7])
            eighth_board = board.t
        else:
            eighth_board = torch.zeros_like(first_board)
        if len(moves) >= 9:
            board.move(moves[8])
            ninth_board = board.t
        else:
            ninth_board = torch.zeros_like(first_board)
        if len(moves) >= 10:
            board.move(moves[9])
            tenth_board = board.t
        else:
            tenth_board = torch.zeros_like(first_board)

        # Convert theme string to one-hot encoding
        themes = [0.0 for _ in THEME_SET]
        puzzle_themes = self.puzzles.iloc[idx, 4].split()  # Assuming themes are in column 4
        for theme in puzzle_themes:
            if theme in THEME2IDX:
                themes[THEME2IDX[theme]] = 1.0

        moves_bucket = get_move_bucket(len(moves))

        return (
            first_board,
            second_board,
            third_board,
            fourth_board,
            fifth_board,
            sixth_board,
            seventh_board,
            eighth_board,
            ninth_board,
            tenth_board,
            len(moves) >= 3,
            len(moves) >= 4,
            len(moves) >= 5,
            len(moves) >= 6,
            len(moves) >= 7,
            len(moves) >= 8,
            len(moves) >= 9,
            len(moves) >= 10,
            torch.tensor(themes, dtype=torch.float32),
            torch.tensor(moves_bucket, dtype=torch.int64),
            None,  # Placeholder for compatibility
            None,  # Placeholder for compatibility
        )


def load_data(
    dataset_name,
    regularize_train_dataset=False,
    flip_boards=0.0,
    use_cleaned_dataset=False,
):
    if dataset_name == "train":
        return SimpleBaselineDataset(
            (
                "datasets/cleaned_train_set.csv"
                if use_cleaned_dataset
                else "datasets/train_set.csv"
            ),
            regularize=regularize_train_dataset,
            flip_boards=flip_boards,
        )
    elif dataset_name == "valid":
        return SimpleBaselineDataset(
            (
                "datasets/cleaned_validation_set.csv"
                if use_cleaned_dataset
                else "datasets/validation_set.csv"
            ),
            flip_boards=flip_boards,
        )
    elif dataset_name == "test":
        return SimpleBaselineDataset(
            (
                "datasets/cleaned_test_set.csv"
                if use_cleaned_dataset
                else "datasets/test_set.csv"
            ),
            flip_boards=flip_boards,
        )
    elif dataset_name == "ieee":
        return IEEEDataset(
            "datasets/ieee_dataset.csv",
            # flip_boards=flip_boards
        )
    else:
        raise ValueError(
            f"invalid dataset name {dataset_name}, must be [train, valid, test, ieee]"
        )


def load_ieee(flip_boards=0.0):
    return IEEEDataset("datasets/ieee_dataset.csv")
    # return IEEEDataset("datasets/ieee_dataset.csv", flip_boards=flip_boards)


class IEEEDataset(Dataset):
    def __init__(self, puzzles_file, filter=False, regularize=False):
        self.puzzles = pd.read_csv(puzzles_file)

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        fen = self.puzzles.iloc[idx, 1]
        moves = self.puzzles.iloc[idx, 2]
        moves = moves.split()

        board = utils.ChessBoard(fen)
        board.move(moves[0])
        first_board = board.t
        board.move(moves[1])
        second_board = board.t

        if len(moves) >= 3:
            board.move(moves[2])
            third_board = board.t
        else:
            third_board = torch.zeros_like(first_board)
        if len(moves) >= 4:
            board.move(moves[3])
            fourth_board = board.t
        else:
            fourth_board = torch.zeros_like(first_board)
        if len(moves) >= 5:
            board.move(moves[4])
            fifth_board = board.t
        else:
            fifth_board = torch.zeros_like(first_board)
        if len(moves) >= 6:
            board.move(moves[5])
            sixth_board = board.t
        else:
            sixth_board = torch.zeros_like(first_board)
        # Moves 7-10
        if len(moves) >= 7:
            board.move(moves[6])
            seventh_board = board.t
        else:
            seventh_board = torch.zeros_like(first_board)
        if len(moves) >= 8:
            board.move(moves[7])
            eighth_board = board.t
        else:
            eighth_board = torch.zeros_like(first_board)
        if len(moves) >= 9:
            board.move(moves[8])
            ninth_board = board.t
        else:
            ninth_board = torch.zeros_like(first_board)
        if len(moves) >= 10:
            board.move(moves[9])
            tenth_board = board.t
        else:
            tenth_board = torch.zeros_like(first_board)

        # Convert theme string to one-hot encoding
        themes = [0.0 for _ in THEME_SET]
        puzzle_themes = self.puzzles.iloc[idx, 4].split() if not pd.isna(self.puzzles.iloc[idx, 4]) else []
        for theme in puzzle_themes:
            if theme in THEME2IDX:
                themes[THEME2IDX[theme]] = 1.0

        moves_bucket = get_move_bucket(len(moves))

        return (
            first_board,
            second_board,
            third_board,
            fourth_board,
            fifth_board,
            sixth_board,
            seventh_board,
            eighth_board,
            ninth_board,
            tenth_board,
            len(moves) >= 3,
            len(moves) >= 4,
            len(moves) >= 5,
            len(moves) >= 6,
            len(moves) >= 7,
            len(moves) >= 8,
            len(moves) >= 9,
            len(moves) >= 10,
            torch.tensor(themes, dtype=torch.float32),
            torch.tensor(moves_bucket, dtype=torch.int64),
            None,  # Placeholder for compatibility
            None,  # Placeholder for compatibility
        )
