import itertools
from .card import Card
from .hand_type import HandType

def generate_singles(hand):
    return [[card] for card in hand]

def generate_pairs(hand):
    pairs = []
    rank_groups = {}
    for card in hand:
        rank_groups.setdefault(card.rank, []).append(card)
    for cards in rank_groups.values():
        if len(cards) >= 2:
            pairs.extend(list(itertools.combinations(cards, 2)))
    return [list(pair) for pair in pairs]

def generate_triples(hand):
    triples = []
    rank_groups = {}
    for card in hand:
        rank_groups.setdefault(card.rank, []).append(card)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            triples.extend(list(itertools.combinations(cards, 3)))
    return [list(triple) for triple in triples]

def generate_five_card_hands(hand):
    five_card_hands = []
    five_card_combos = list(itertools.combinations(hand, 5))
    for combo in five_card_combos:
        hand_type, key = evaluate_five_card_hand(list(combo))
        if hand_type != HandType.INVALID:
            five_card_hands.append((list(combo), hand_type, key))
    return five_card_hands


def compare_hands(hand1, hand2):
    type1, key1 = hand1
    type2, key2 = hand2

    if type1 != type2:
        return type1.value - type2.value  # Higher hand type wins

    # For same hand types, compare based on keys
    if type1 in [HandType.SINGLE, HandType.PAIR, HandType.TRIPLE, HandType.STRAIGHT, HandType.FLUSH, HandType.STRAIGHT_FLUSH]:
        return compare_cards(key1, key2)
    elif type1 == HandType.FULL_HOUSE:
        # Compare triple ranks
        return key1 - key2
    elif type1 == HandType.FOUR_OF_A_KIND:
        # Compare four ranks
        four1_rank, kicker1 = key1
        four2_rank, kicker2 = key2
        if four1_rank != four2_rank:
            return four1_rank - four2_rank
        else:
            return compare_cards(kicker1, kicker2)
    else:
        return 0  # Should not occur

def compare_cards(card1, card2):
    rank1 = Card.rank_order[card1.rank]
    rank2 = Card.rank_order[card2.rank]
    if rank1 != rank2:
        return rank1 - rank2
    else:
        # If ranks are equal, compare suits
        suit1 = Card.suit_order[card1.suit]
        suit2 = Card.suit_order[card2.suit]
        return suit1 - suit2
    
def is_valid_hand(cards):
    if not cards:
        return HandType.INVALID, None
    if len(cards) == 1:
        return HandType.SINGLE, cards[0]
    elif len(cards) == 2:
        return is_valid_pair(cards)
    elif len(cards) == 3:
        return is_valid_triple(cards)
    elif len(cards) == 5:
        return evaluate_five_card_hand(cards)
    else:
        return HandType.INVALID, None

def is_valid_pair(cards):
    if cards[0].rank == cards[1].rank:
        return HandType.PAIR, max(cards)
    return HandType.INVALID, None

def is_valid_triple(cards):
    if cards[0].rank == cards[1].rank == cards[2].rank:
        return HandType.TRIPLE, max(cards)
    return HandType.INVALID, None


def evaluate_five_card_hand(cards):
    cards_sorted = sorted(cards)
    ranks = [Card.rank_order[card.rank] for card in cards_sorted]
    suits = [card.suit for card in cards_sorted]
    unique_ranks = set(ranks)
    unique_suits = set(suits)
    is_flush = len(unique_suits) == 1
    is_straight = ranks == list(range(ranks[0], ranks[0] + 5))

    rank_count = {}
    for rank in ranks:
        rank_count[rank] = rank_count.get(rank, 0) + 1
    count_values = sorted(rank_count.values())

    if is_straight and is_flush:
        return HandType.STRAIGHT_FLUSH, cards_sorted[-1]
    elif count_values == [1, 4]:
        # Four of a Kind
        four_rank = max(rank_count, key=lambda k: rank_count[k])
        kicker = next(card for card in cards if Card.rank_order[card.rank] != four_rank)
        return HandType.FOUR_OF_A_KIND, (four_rank, kicker)
    elif count_values == [2, 3]:
        # Full House
        triple_rank = max(rank_count, key=lambda k: (rank_count[k], k))
        return HandType.FULL_HOUSE, triple_rank
    elif is_flush:
        return HandType.FLUSH, cards_sorted[-1]
    elif is_straight:
        return HandType.STRAIGHT, cards_sorted[-1]
    else:
        return HandType.INVALID, None