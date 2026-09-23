"""Compute dataset facts rather than relying on the task's summary statistics."""

from collections import Counter
from datetime import timedelta
import json

from eventmatch.catalog import ROOT, load_catalog
from eventmatch.models import START, END


def audit(catalog):
    profiles = catalog.profiles
    cities = sorted({p.city for p in profiles})
    categories = sorted({c for p in profiles for c in p.categories})
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    monthly = {}
    for month in (9, 10, 11, 12):
        month_days = [d for d in days if d.month == month]
        weekends = [d for d in month_days if d.weekday() >= 5]
        occupied = sum(d.isoformat() in p.busy_dates for p in profiles for d in month_days)
        weekend_occupied = sum(d.isoformat() in p.busy_dates for p in profiles for d in weekends)
        monthly[f"2026-{month:02}"] = {
            "covered_days": len(month_days), "busy_profile_days": occupied,
            "total_profile_days": len(profiles) * len(month_days),
            "occupancy_percent": round(100 * occupied / (len(profiles) * len(month_days)), 2),
            "weekend_occupancy_percent": round(100 * weekend_occupied / (len(profiles) * len(weekends)), 2),
        }
    return {
        "dataset_sha256": catalog.sha256,
        "profiles": len(profiles), "categories": len(categories),
        "cities": dict(sorted(Counter(p.city for p in profiles).items())),
        "flags": {key: sum(getattr(p, key) for p in profiles)
                  for key in ("synthetic", "city_imputed", "price_imputed")},
        "non_synthetic_profiles": sum(not p.synthetic for p in profiles),
        "null_max_hours": sum(p.max_hours is None for p in profiles),
        "multi_category_profiles": sum(len(p.categories) > 1 for p in profiles),
        "price_min_kzt": min(p.price_from_kzt for p in profiles),
        "price_max_kzt": max(p.price_from_kzt for p in profiles),
        "category_city_matrix": {
            category: {**{city: sum(category in p.categories and p.city == city for p in profiles)
                           for city in cities},
                       "total": sum(category in p.categories for p in profiles)}
            for category in categories
        },
        "monthly_occupancy": monthly,
    }


def main():
    result = audit(load_catalog())
    target = ROOT / "data" / "reports" / "audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
