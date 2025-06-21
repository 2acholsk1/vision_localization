from src.matchers.lbp_matcher import MatcherLBP
from src.matchers.lbp_pt_matcher import MatcherPyTorchLBP
from src.matchers.nn_matcher import NNMatcher
from src.resamplers.bootstrap_resampler import BootstrapResampler
from src.resamplers.deterministic_resampler import DeterministicResampler
from src.resamplers.multinomial_resampler import MultinomialResampler
from src.resamplers.rejection_resampler import RejectionResampler
from src.resamplers.residual_resampler import ResidualResampler
from src.resamplers.restricted_resampler import RestrictedResampler
from src.resamplers.straticfied_resampler import StratifiedResampler
from src.resamplers.systematic_resampler import SystematicResampler


def choose_matcher(matcher_name, encoder_name, embedding_size, weights_path):
    match(matcher_name):
        case 'LBP':
            return MatcherLBP()
        case 'LBP_PT':
            return MatcherPyTorchLBP()
        case 'NN':
            return NNMatcher(encoder_name, embedding_size, weights_path)


def choose_resampler(resampler_name, number):
    match(resampler_name):
        case 'Bootstrap':
            return BootstrapResampler(number)
        case 'Determinitstic':
            return DeterministicResampler(number)
        case 'Multinomial':
            return MultinomialResampler(number)
        case 'Rejection':
            return RejectionResampler(number)
        case 'Resiudal':
            return ResidualResampler(number)
        case 'Restricted':
            return RestrictedResampler(number)
        case 'Stratified':
            return StratifiedResampler(number)
        case 'Systematic':
            return SystematicResampler(number)
