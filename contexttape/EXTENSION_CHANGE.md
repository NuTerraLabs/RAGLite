# File Extension Change: .ts → .is

## Summary

Changed all segment file extensions from `.ts` to `.is` to avoid confusion with TypeScript files.

## What Changed

### Source Code (src/contexttape/)
- ✅ storage.py - segment path, glob pattern, docstrings
- ✅ cli.py - all file path references in output
- ✅ search.py - file paths in search results
- ✅ relevance.py - file paths in relevance output
- ✅ __init__.py - package docstring

### Documentation
- ✅ All README files
- ✅ WHAT_YOU_GET.md
- ✅ SIMPLE_GUIDE.md
- ✅ PROJECT_STRUCTURE.md
- ✅ ORGANIZATION.md
- ✅ All other .md files

### Test Scripts
- ✅ test_system.py - updated to look for .is files

## New Behavior

**Before:**
```
data/my_store/
├── segment_0.ts
├── segment_1.ts
└── segment_2.ts
```

**After:**
```
data/my_store/
├── segment_0.is
├── segment_1.is
└── segment_2.is
```

## API - No Changes

The API remains exactly the same:

```python
from contexttape import ISStore

store = ISStore("data/my_store")  # Creates .is files now
store.append_text("text")
store.append_vector_i8(embedding)
```

## File Format - Unchanged

The file format is identical:
- 32-byte header
- Encrypted payload
- Same data structures

Only the file extension changed: `.ts` → `.is`

## Testing

✅ All tests pass with new `.is` extension  
✅ Files are created correctly  
✅ Search works as expected  
✅ Multi-store works  

Run `./run_test.sh` to verify.
