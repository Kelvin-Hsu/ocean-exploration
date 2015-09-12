from computers import gp
import logging

def predictions(learned_classifier, Fqw, fusemethod = 'EXCLUSION'):

    y_unique = learned_classifier[0].cache.get('y_unique') # Always multiclass
    logging.info('Caching Predictor...')
    predictor = gp.classifier.query(learned_classifier, Fqw)
    logging.info('Computing Expectance...')
    fq_exp = gp.classifier.expectance(learned_classifier, predictor)
    logging.info('Computing Variance...')
    fq_var = gp.classifier.variance(learned_classifier, predictor)
    logging.info('Computing Prediction Probabilities...')
    yq_prob = gp.classifier.predict_from_latent(fq_exp, fq_var, 
        learned_classifier, fusemethod = fusemethod)
    logging.info('Computing Prediction...')
    yq_pred = gp.classifier.classify(yq_prob, y_unique)
    logging.info('Computing Prediction Information Entropy...')
    yq_mie = gp.classifier.entropy(yq_prob)    
    logging.info('Computing Linearised Model Differential Entropy...')
    yq_lmde = gp.classifier.linearised_model_differential_entropy(
        fq_exp, fq_var, learned_classifier)
    return yq_pred, yq_mie, yq_lmde

def miss_ratio(yq_pred, yq_truth):
    return (yq_pred - yq_truth).nonzero()[0].shape[0] / yq_truth.shape[0]