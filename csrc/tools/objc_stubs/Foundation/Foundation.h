// Minimal Foundation surface for off-macOS syntax checking only.
// Declares just the members csrc/**/*.mm touches, so clang can parse the
// Objective-C++ on Linux. Never compiled or linked on macOS -- the real SDK
// headers win there. Add members as the sources need them.
#pragma once
#include <cstddef>
#if !defined(nil)
#define nil ((id)0)
#endif
#if !defined(YES)
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif
typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;
@class NSString;
@interface NSObject
+ (instancetype)alloc;
- (instancetype)init;
- (instancetype)self;
@end
@interface NSString : NSObject
@property(readonly) const char* UTF8String;
- (BOOL)isEqualToString:(NSString*)other;
+ (instancetype)stringWithUTF8String:(const char*)c;
@end
@interface NSError : NSObject
@property(readonly) NSString* domain;
@property(readonly) NSInteger code;
@property(readonly) NSString* localizedDescription;
@end
@interface NSArray<__covariant T> : NSObject
@end
@interface NSMutableArray<T> : NSArray <T>
- (void)addObject:(T)o;
- (void)removeAllObjects;
@property(readonly) NSUInteger count;
- (T)objectAtIndexedSubscript:(NSUInteger)i;
- (T)objectAtIndex:(NSUInteger)i;
- (void)replaceObjectAtIndex:(NSUInteger)i withObject:(T)o;
+ (instancetype)array;
@end
@interface NSDictionary<__covariant K, __covariant V> : NSObject
@property(readonly) NSUInteger count;
@end
@interface NSMutableDictionary<K, V> : NSDictionary <K, V>
- (V)objectForKeyedSubscript:(K)k;
- (void)setObject:(V)v forKeyedSubscript:(K)k;
- (V)objectForKey:(K)k;
- (void)setObject:(V)v forKey:(K)k;
- (void)removeAllObjects;
+ (instancetype)dictionary;
@end

@interface NSURL : NSObject
+ (instancetype)fileURLWithPath:(NSString*)p;
@end
