# Submission Package — Text Fields

Use this content directly for the HackerEarth/portal submission form fields.

---

## Title

**ParkWatch BLR: AI-Driven Parking Hotspot Detection & Congestion Impact Scoring**

(Alternative, shorter): **ParkWatch BLR — Parking Intelligence for Targeted Traffic Enforcement**

---

## Description

Use this as the main submission description (trim to fit character limits if
the portal has one — paragraph order is written so it still works if you cut
from the bottom):

> Illegal and spillover parking near commercial areas, metro stations, and
> events chokes Bengaluru's carriageways, but enforcement today is
> patrol-based and reactive — there is no way to see, at a city-wide level,
> which violations are actually causing the worst congestion, or where to
> send limited enforcement resources first.
>
> ParkWatch BLR processes 298,450 anonymized violation records from
> Bengaluru Traffic Police (Nov 2023–Apr 2024) into an actionable
> enforcement-priority system. Since the raw data has no road speed or
> traffic-flow measurements, we built a transparent Congestion Impact Score
> (CIS) for every violation — a documented function of violation severity
> (double parking and main-road parking score far higher than footpath
> parking), vehicle footprint (a parked bus blocks more lane width than a
> scooter), and time-of-day traffic weight (peak-hour obstruction matters
> more than a 3am one).
>
> We grid the city into 150m enforcement zones, score each by an
> Enforcement Priority Index combining violation volume, day-to-day
> persistence, and severity ceiling, and rank all 2,293 qualifying zones —
> surfacing 687 Critical/High priority hotspots. A Random Forest model
> (R² = 0.91) then learns the underlying spatial-temporal risk pattern,
> letting the system forecast expected congestion impact for any
> location/time/vehicle combination, not just report history.
>
> The result is an interactive dashboard — map, searchable leaderboard,
> and trend charts — that lets a traffic planner answer, in seconds,
> "where should today's enforcement patrol go, and why."
>
> We've been transparent about what this approach can and cannot claim:
> CIS is a documented proxy, not measured congestion; ~75% of detections
> occur at night because that is when enforcement devices are active, not
> necessarily when illegal parking peaks; and we discarded density-based
> clustering after it produced a single cluster covering 55% of all
> citywide violations, replacing it with stable fixed-grid zoning. Full
> methodology, all weights, and known limitations are documented in the
> repository.

---

## Snapshots — what to capture

Take 4–6 screenshots from the dashboard (`dashboard/index.html`, opened in a
normal browser with internet access so map tiles load). Suggested shot list,
in the order judges will likely scroll through them:

1. **Full dashboard top view** — header + stat cards + map + leaderboard
   side by side. This is the "first impression" shot; make sure the map has
   fully loaded with visible Bengaluru streets and colored hotspot markers
   before capturing.
2. **Map close-up with a popup open** — click a Critical-tier marker (e.g.
   the #1 ranked zone) so its popup is visible, showing EPI, violation
   count, dominant violation/vehicle type. This proves the system gives
   actionable per-zone detail, not just a heatmap.
3. **Leaderboard with search/filter in use** — type a police station name
   into the search box or select "Critical" in the tier filter, so the
   table visibly updates. Shows the tool is interactive, not a static
   report.
4. **Hour-of-day chart** — the temporal chart showing the night-skew in
   red. This screenshot directly supports the "we found and corrected for
   a bias in the data" narrative, which is a strong differentiator.
5. **Violation/vehicle breakdown charts** — shows the system understands
   violation taxonomy and isn't a black box.
6. **Methodology panel** — the three-caveat panel at the bottom. Judges
   evaluating technical rigor specifically look for whether a team
   understands their own data's limitations; this screenshot answers that
   proactively.

Capture at a reasonably wide browser window (1440px or wider) so text isn't
cramped. On most systems: open the dashboard, press F11 for fullscreen or
just maximize the window, then use your OS screenshot tool (Windows: Win+Shift+S,
Mac: Cmd+Shift+4, or any browser extension) for each shot.

---

## Video URL — what to record and how

Most hackathon portals want a short (2–4 minute) walkthrough video, hosted
on YouTube (can be Unlisted) or a Drive link.

**Suggested script / shot order (aim for ~3 minutes total):**

1. **0:00–0:20 — Problem framing.** State the problem in your own words on
   camera or as a title slide: "Illegal parking chokes Bengaluru traffic,
   but enforcement is reactive and there's no way to prioritize where to
   act." Show the original problem statement briefly.
2. **0:20–0:50 — Data and methodology, briefly.** Show the README or
   `metadata.json` for two seconds, then explain in plain language: "We
   built a Congestion Impact Score because the data has no traffic speed
   measurements — here's how it's computed" (show the CIS formula on
   screen, just the one line, don't read code).
3. **0:50–2:00 — Live dashboard walkthrough.** This is the core of the
   video. Open `dashboard/index.html`, and:
   - Pan/zoom the map, click 2-3 markers showing different tiers
   - Use the search bar to filter the leaderboard by a station name
   - Point out the hour-of-day chart and explicitly call out the
     night-skew finding — this is your strongest "we understood our data"
     moment, don't skip it
   - Show the methodology/limitations panel and say one sentence about
     why grid zoning was chosen over clustering
4. **2:00–2:30 — Predictive model.** Mention the Random Forest model and
   its R² = 0.91 — say what it lets you do that a static heatmap can't
   (forecast risk for unobserved location/time combinations).
5. **2:30–3:00 — Close.** Restate the impact: "this turns 300,000 raw
   violation logs into a ranked, explainable list of where enforcement
   should go next." Mention this is PS1 specifically and that it directly
   produces the deliverable the problem statement asked for (hotspot
   detection + quantified congestion impact).

**Recording tools:** screen recording can be done with OBS Studio (free,
cross-platform), the built-in Windows Game Bar (Win+G), macOS QuickTime
screen recording, or Loom (free tier covers this length). Record at 1080p
minimum. Keep narration conversational — reading a script verbatim usually
sounds worse than explaining it naturally once you know the beats above.

**Where to host:** YouTube (set to "Unlisted," not "Private," so the link
works without requiring viewers to request access), or Google Drive with
sharing set to "Anyone with the link can view." Test the link in an
incognito/private browser window before submitting to confirm a reviewer
with no prior access can actually open it.

---

## Repository link checklist

Before submitting the GitHub link, confirm:

- [ ] `README.md` is at the repo root and renders correctly on GitHub
  (check tables and code blocks display properly)
- [ ] `pipeline.py` runs cleanly on a fresh clone (test in a new virtualenv
  if possible)
- [ ] `dashboard/index.html` opens correctly by double-click with no
  console errors (vendor/ folder must be committed, not gitignored —
  double check `.gitignore` doesn't accidentally exclude `vendor/*.js`)
- [ ] The raw dataset is either committed (if under GitHub's file size
  limits — check, it's ~110MB which exceeds GitHub's default 100MB
  warning threshold) or excluded with clear instructions in the README
  on where to place it, plus a small sample CSV committed instead for
  quick inspection
- [ ] No API keys, credentials, or personal file paths (e.g.
  `C:\Users\LAKSHMI\Flip`) appear anywhere in committed files
- [ ] Commit history doesn't expose the raw uploaded dataset before it was
  meant to be public, if anonymization/licensing terms restrict redistribution
  — check the HackerEarth dataset's usage terms before pushing it publicly
