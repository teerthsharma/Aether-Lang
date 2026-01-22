# 🦭 Learning AEGIS - The Fun Way!

Welcome to AEGIS! This guide will teach you how to code in AEGIS, step by step. Even if you've never coded before, you'll be creating cool 3D visualizations by the end! 🚀

---

## 📖 Table of Contents

1. [What is AEGIS?](#what-is-aegis)
2. [Your First Program](#your-first-program)
3. [Variables - Storing Stuff](#variables---storing-stuff)
4. [Math Operations](#math-operations)
5. [Making Decisions with If](#making-decisions-with-if)
6. [The Magic Seal Loop 🦭](#the-magic-seal-loop-)
7. [Functions - Reusable Code](#functions---reusable-code)
8. [3D Manifolds - The Cool Part!](#3d-manifolds---the-cool-part)
9. [Cheat Sheet](#cheat-sheet)

---

## What is AEGIS?

AEGIS is a programming language that lets you:
- ✨ Turn data into 3D shapes
- 🔄 Make loops that know when to stop by themselves
- 📊 Train AI models visually

**Special feature:** Every statement ends with a `~` tilde!

---

## Your First Program

Let's start simple. Here's how to say "Hello" in AEGIS:

```aegis
print("Hello, World!")~
```

**What happens?** → The computer shows: `Hello, World!`

### Try more:

```aegis
print("My name is AEGIS!")~
print("I can do math:", 2 + 2)~
print("🦭 Seal says hi!")~
```

**Remember:** Every line ends with `~`

---

## Variables - Storing Stuff

A **variable** is like a labeled box where you store things.

```aegis
// Create a box called "age" and put 12 in it
let age = 12~

// Create a box called "name" and put your name in it
let name = "Alex"~

// Show what's in the boxes
print("I am", name)~
print("I am", age, "years old")~
```

### Rules for Variable Names:
- ✅ Use letters, numbers, and underscores: `my_score`, `player1`
- ❌ Don't start with a number: `1player` is wrong
- ❌ Don't use spaces: `my score` is wrong, use `my_score`

---

## Math Operations

AEGIS can do all kinds of math!

| Symbol | What it does | Example | Result |
|--------|--------------|---------|--------|
| `+` | Add | `5 + 3` | `8` |
| `-` | Subtract | `10 - 4` | `6` |
| `*` | Multiply | `6 * 7` | `42` |
| `/` | Divide | `20 / 4` | `5` |
| `%` | Remainder | `10 % 3` | `1` |

### Example:

```aegis
let apples = 10~
let friends = 3~
let each = apples / friends~
let leftover = apples % friends~

print("Each friend gets", each, "apples")~
print("Leftovers:", leftover)~
```

---

## Making Decisions with If

Sometimes you want the computer to make choices. Use `if`!

```aegis
let score = 85~

if score >= 90 {
    print("Amazing! A grade!")~
} else if score >= 80 {
    print("Great! B grade!")~
} else if score >= 70 {
    print("Good! C grade!")~
} else {
    print("Keep trying!")~
}
```

### Comparison Symbols:

| Symbol | Meaning |
|--------|---------|
| `==` | Equal to |
| `!=` | Not equal to |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

---

## The Magic Seal Loop 🦭

Here's what makes AEGIS special - the **seal loop**! 

A seal loop is like a smart helper that keeps working until the job is "sealed" (finished perfectly).

### Basic Seal Loop:

```aegis
let count = 0~

🦭 until count >= 5 {
    print("Count is:", count)~
    count = count + 1~
}
print("Done! The loop is sealed!")~
```

**Output:**
```
Count is: 0
Count is: 1
Count is: 2
Count is: 3
Count is: 4
Done! The loop is sealed!
```

### Why is it called "seal"? 🦭

1. **Like a wax seal** - It seals (closes) when the work is complete
2. **Like the animal** - Seals are smart and efficient!
3. **The emoji** - Because coding should be fun! 🦭

### Seal Loop for Counting:

```aegis
// Count from 1 to 10
seal for i in 1..11 {
    print(i)~
}
```

---

## Functions - Reusable Code

A **function** is like a recipe - you write it once, use it many times!

### Making a Function:

```aegis
fn say_hello(name) {
    print("Hello,", name, "!")~
}

// Now use it!
say_hello("Alice")~
say_hello("Bob")~
say_hello("Charlie")~
```

### Function that Returns a Value:

```aegis
fn add_numbers(a, b) {
    return a + b~
}

let result = add_numbers(5, 3)~
print("5 + 3 =", result)~  // Shows: 5 + 3 = 8
```

---

## 3D Manifolds - The Cool Part!

This is the superpower of AEGIS - turning data into 3D shapes!

### Step 1: Create Data

```aegis
let temps = [20, 22, 25, 23, 21, 19, 24, 26, 25, 22]~
```

### Step 2: Turn it into a 3D Shape

```aegis
manifold Weather = embed(temps, dim=3, tau=2)~
```

### Step 3: See It!

```aegis
// Show as ASCII art in terminal
render Weather { format: "ascii" }~

// Or export to view in browser
render Weather { format: "webgl", output: "weather.html" }~
```

---

## Cheat Sheet

### Quick Reference

| What | How to Write | Example |
|------|--------------|---------|
| Print | `print(...)~` | `print("Hi!")~` |
| Variable | `let name = value~` | `let x = 10~` |
| Add | `+` | `5 + 3` |
| Subtract | `-` | `10 - 4` |
| Multiply | `*` | `6 * 7` |
| Divide | `/` | `20 / 4` |
| If | `if condition { }` | `if x > 5 { }` |
| Seal loop | `🦭 until { }` | `🦭 until x > 10 { }` |
| For loop | `seal for i in start..end { }` | `seal for i in 0..10 { }` |
| Function | `fn name() { }` | `fn greet() { }` |
| Return | `return value~` | `return 42~` |
| Comment | `// text` | `// This is ignored` |

### Special Symbols

| Symbol | Name | Used For |
|--------|------|----------|
| `~` | Tilde | End every statement! |
| `🦭` | Seal emoji | Seal loops! |
| `{ }` | Curly braces | Grouping code |
| `( )` | Parentheses | Function calls |
| `[ ]` | Square brackets | Lists/arrays |
| `..` | Range | `0..10` means 0 to 9 |
| `//` | Comment | Notes for humans |

---

## 🎯 Practice Challenges

Try these on your own!

### Challenge 1: Countdown
Make a program that counts down from 10 to 1, then says "Blast off! 🚀"

### Challenge 2: Times Table
Write a function that prints the times table for any number.

### Challenge 3: 3D Star
Create a manifold that looks like a star shape!

---

<div align="center">

**Happy Coding! 🦭✨**

*Made with ❤️ by the AEGIS Team*

</div>
