"""Agent tools. The model may select sources, never select/reorder contractors."""
from copy import deepcopy
from dataclasses import replace
import secrets
from time import monotonic

from .engine import recommend, normalize, failures, hits, money, POLICY_VERSION
from .models import Request, START, END, FORMATS, LANGUAGES, iso_date
from .semantic import SemanticRanker, fragments
from .store import Store

SERVICE_VERSION = 'eventmatch-agent-v1'


class Service:
    def __init__(self, catalog, settings, store=None):
        self.catalog, self.settings = catalog, settings
        self.by_id = {p.id: p for p in catalog.profiles}
        self.ranker = (SemanticRanker(catalog, settings.artifact, settings.nvidia_model)
                       if settings.ranker == 'nvidia' else None)
        self.store = store or Store(settings.database)

    def get_catalog_metadata(self):
        result = {'cities': sorted({p.city for p in self.catalog.profiles}),
                  'categories': sorted({c for p in self.catalog.profiles for c in p.categories}),
                  'event_formats': FORMATS, 'languages': LANGUAGES,
                  'calendar_start': str(START), 'calendar_end': str(END),
                  'catalog_version': self.catalog.sha256, 'service_version': SERVICE_VERSION,
                  'ranking_mode': 'nvidia_semantic_artifact' if self.ranker else 'deterministic_baseline',
                  'nvidia': self.ranker.metadata if self.ranker else {'usage': 'not_used'},
                  'openai_configured': bool(self.settings.openai_key)}
        self.store.audit({'tool': 'get_catalog_metadata', 'arguments': {}, 'metadata': result})
        return result

    def validate(self, raw):
        request = raw if isinstance(raw, Request) else Request.parse(raw)
        cities = {normalize(p.city): p.city for p in self.catalog.profiles}
        categories = {normalize(c): c for p in self.catalog.profiles for c in p.categories}
        if normalize(request.city) not in cities:
            raise ValueError('city: город отсутствует в metadata каталога')
        if normalize(request.category) not in categories:
            raise ValueError('category: категория отсутствует в metadata каталога')
        return replace(request, city=cities[normalize(request.city)],
                       category=categories[normalize(request.category)])

    def recommend(self, request):
        started = monotonic()
        request = self.validate(request)
        result = recommend(self.catalog, request, self.ranker)
        result['result_id'] = secrets.token_urlsafe(24)
        result['normalized_request'] = result['request']
        result['metadata'] = result['meta']
        result['metadata'].update(service_version=SERVICE_VERSION,
            ranking_mode='nvidia_semantic_artifact' if self.ranker else 'deterministic_baseline',
            nvidia=self.ranker.metadata if self.ranker else {'usage': 'not_used'},
            openai={'used': False}, explanation_mode='template_explanation_fallback')
        for card in result['cards']:
            p = self.by_id[card['id']]
            chunks = fragments(p)
            if self.ranker:
                chunks = self.ranker.evidence(p, request)
                card['ranking'] = {'semantic_score_int': self.ranker.score(p, request),
                                   'price_tiebreak_kzt': p.price_from_kzt, 'id_tiebreak': p.id}
            else:
                chunks = sorted(chunks, key=lambda c: (-len(hits(c['quote'], request.event_format)),
                                                      c['start']))
            card['available_evidence'] = chunks[:6]
            values = {
                'format': f'Берёт формат «{request.event_format}»',
                'price': f'цена от {money(p.price_from_kzt)} укладывается в бюджет {money(request.budget_kzt)}',
                'date': f'на {request.event_date} в известном календаре занятость не указана',
                'language': ('языки работы: ' + ', '.join(p.languages)),
                'duration': ('ограничение по часам присутствия неприменимо' if p.max_hours is None
                             else f'максимальное время присутствия — {p.max_hours:g} ч'),
            }
            card['available_facts'] = [{'fact_id': f'{p.id}:{k}', 'profile_id': p.id,
                                        'field': k, 'text': v} for k, v in values.items()]
            # Provenance cannot be inferred from IDs or display names.
            if p.synthetic:
                card['provenance']['labels'][0] = 'Синтетический профиль'
        self.store.put('result', result['result_id'], result)
        result = self.finalize_result(result['result_id'], self.default_plan(result), audit=False)
        self.store.audit({'tool': 'recommend', 'arguments': request.to_dict(),
                          'selected_ids': [c['id'] for c in result['cards']],
                          'diagnostics': result['diagnostics'], 'metadata': result['metadata'],
                          'duration_ms': round((monotonic() - started) * 1000, 2)})
        return result

    @staticmethod
    def default_plan(result):
        return {'cards': [{'profile_id': c['id'],
                           'fact_ids': [c['id'] + ':format', c['id'] + ':price'],
                           'evidence_id': c['available_evidence'][0]['evidence_id']}
                          for c in result['cards']]}

    def finalize_result(self, result_id, explanation_plan, audit=True):
        result = self.store.get('result', result_id)
        request = self.validate(result['normalized_request'])
        fresh = recommend(self.catalog, request, self.ranker)
        if [c['id'] for c in fresh['cards']] != [c['id'] for c in result['cards']]:
            raise ValueError('Версия каталога или ранжирования изменилась; повторите подбор')
        if (not isinstance(explanation_plan, dict) or set(explanation_plan) != {'cards'}
                or not isinstance(explanation_plan['cards'], list)):
            raise ValueError('Неверный план объяснений')
        plan = explanation_plan['cards']
        if (len(plan) != len(result['cards']) or any(not isinstance(p, dict) for p in plan)
                or [p.get('profile_id') for p in plan] != [c['id'] for c in result['cards']]):
            raise ValueError('План должен сохранять ID, количество и порядок карточек')
        for selection, card in zip(plan, result['cards']):
            if set(selection) != {'profile_id', 'fact_ids', 'evidence_id'}:
                raise ValueError('Неизвестные поля плана')
            facts = {f['fact_id']: f for f in card['available_facts']}
            evidence = {e['evidence_id']: e for e in card['available_evidence']}
            ids = selection['fact_ids']
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 3
                    or any(not isinstance(i, str) or i not in facts for i in ids)
                    or len(ids) != len(set(ids))
                    or not isinstance(selection['evidence_id'], str)
                    or selection['evidence_id'] not in evidence):
                raise ValueError('Факт или цитата не принадлежат карточке')
            quote = evidence[selection['evidence_id']]
            profile = self.by_id[card['id']]
            if (quote['profile_id'] != profile.id
                    or profile.description[quote['start']:quote['end']] != quote['quote']
                    or failures(profile, request)):
                raise ValueError('Не удалось проверить источник или условия')
            card['explanation'] = ('; '.join(facts[i]['text'] for i in ids)
                                   + '. В описании профиля: «' + quote['quote'] + '».')
            card['selected_sources'] = deepcopy(selection)
            card['evidence'] = [e for e in card['evidence'] if e['field'] != 'description'] + [quote]
        result['meta'] = result['metadata']
        self.store.put('result', result_id, result)
        if audit:
            self.store.audit({'tool': 'finalize_result', 'arguments': {'result_id': result_id,
                              'explanation_plan': explanation_plan}, 'metadata': result['metadata']})
        return result

    def compare_dates(self, request, date_a, date_b):
        iso_date(date_a, 'date_a')
        iso_date(date_b, 'date_b')
        request = self.validate({**request, 'event_date': date_a})
        a = self.recommend(request)
        b = self.recommend(replace(request, event_date=date_b))
        ids_a = {c['id'] for c in a['cards']}
        ids_b = {c['id'] for c in b['cards']}
        changes = []
        for p in self.catalog.profiles:
            if p.city != request.city or request.category not in p.categories:
                continue
            reasons_a = failures(p, request)
            reasons_b = failures(p, replace(request, event_date=date_b))
            if reasons_a != reasons_b or p.id in ids_a ^ ids_b:
                code = ('calendar_changed' if reasons_a != reasons_b
                        else 'top3_competition' if not reasons_a else 'other_constraints')
                changes.append({'profile_id': p.id, 'reason': code,
                                'reasons_a': reasons_a, 'reasons_b': reasons_b,
                                'eligible_a': not reasons_a, 'eligible_b': not reasons_b,
                                'shown_a': p.id in ids_a, 'shown_b': p.id in ids_b})
        result = {'status': 'compared', 'date_a': date_a, 'date_b': date_b,
                  'results': [a, b], 'appeared_ids': sorted(ids_b - ids_a),
                  'disappeared_ids': sorted(ids_a - ids_b), 'changes': changes,
                  'summary': 'Сравнение двух дат при одинаковых остальных условиях.'}
        self.store.audit({'tool': 'compare_dates', 'arguments': {'request': request.to_dict(),
                          'date_a': date_a, 'date_b': date_b}, 'changes': changes})
        return result
