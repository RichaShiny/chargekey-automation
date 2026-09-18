# Jurisdiction and schema notes

The source project is intentionally **site-schema aware**. It does not assume that all jurisdictions expose the same charge-key attribute columns.

## Jail jurisdictions

The jail updater supports 16 sites:

- Allegheny
- Buncombe
- Charleston
- Cook
- Harris
- Lucas
- Mecklenburg
- Milwaukee
- Multnomah
- New Orleans
- Palm Beach
- Pennington
- Pima
- San Francisco
- Spokane
- St. Louis

## Court jurisdictions

The court pipeline source defines eight supported downstream sites:

- Allegheny
- Charleston
- Lucas
- Milwaukee
- New Orleans
- Palm Beach
- San Francisco
- Spokane

Court processing runs after jail processing. It uses the finalized jail `Charge Key` as the reference and matches court descriptions to jail descriptions because court and jail charge codes are not directly comparable.

## Site-specific normalization

| Site | Rule |
|---|---|
| Buncombe | Drop rows with missing charge code during preprocessing |
| Spokane | Preserve the merging-guide lowercase description convention; codes are still normalized uppercase by the integrated jail app |
| New Orleans | Accept `NewOrleans` and `New_Orleans` filename prefixes |
| Palm Beach | Accept `PalmBeach`, `Palm_Beach`, and `Palm Beach` prefixes |
| San Francisco | Accept `San Francisco`, `San_Francisco`, and `SanFrancisco` prefixes |
| St. Louis | Accept `StLouis` and `St_Louis` prefixes |

All other jail sites use uppercased descriptions and the normal generated filename prefixes.

Historical columns `chrg_type_FTA` and `chrg_type_FTC` are normalized to `chrg_type_fta` and `chrg_type_ftc`. The helper column `check`, seen in some Spokane keys, is dropped before schema discovery.

## Dynamic attribute schema

For the jail workflow, output attributes are derived from the reference key **for that site at runtime**:

1. include `chrg_ibr_code` only if it exists;
2. include every source column beginning with `chrg_type`;
3. keep the source column order;
4. do not invent missing canonical columns;
5. do not drop extra site-specific `chrg_type*` columns.

The canonical SJC-style attribute set remains an audit reference only. `schemas.audit_jail_schema()` can report missing and extra fields without mutating the site's real schema.

For court output, all `chrg_type*` fields present in the finalized jail key are carried, plus `chrg_ibr_code` and `chrg_sev` when present.

## Output contracts

Jail output keeps the three existing sheets:

- `Charge Key`
- `Charges added`
- `Charge to be Reviewed`

The historical exported column spelling `simiarity_score` is retained for compatibility, while internal code uses `similarity_score`.

Court output keeps:

- `confident`
- `ensemble_resolved`
- `needs_review`
