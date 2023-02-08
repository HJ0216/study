import java.util.*;

public class NumberGame {

	public static void main(String[] args) {
		int com;
		int user;
		int count=0;
		
		com = (int)(Math.random()*100) + 1;
		
		Scanner scan = new Scanner(System.in);
//		user = scan.nextInt(); // 입력받는 곳에 위치
		
		while(true) {
		
		System.out.println("Please enter the number between 1 and 100 (" + com + ") : ");

			while(true) {
				count++; // 1
				
				System.out.println("Enter the number: ");
				user = scan.nextInt(); // 입력받는 곳에 위치
				if(com == user) break;
				else if(com > user) {System.out.println("Upper than " + user);}		
				else {System.out.println("Lower than " + user);}
			} // inner-while
		
		System.out.println("Game Over, tried number: " + count);

		while(true) {
			
			System.out.println("One more try? [y/n]");
	//		int yn = scan.nextInt();
			String yn = scan.next();
			
	//		if(yn == 'n' || yn == 'N') {break;}
			if(yn.equals("n") || yn.equals("N")) {break;}
			// 문자열: ==(주소값), equals(입력값)
			if(yn.equals("y") || yn.equals("Y")) {break;}
			
			} // outter-while
			
			System.out.println("Terminated");


		}
		
//		count		
//
//		while(user!=com) {
//			++count;
//			if(user>com) {
//				System.out.println("The number is less than " + user);
//				user = scan.nextInt();
//			} else {
//				System.out.println("The number is greater than " + user);
//				user = scan.nextInt();
//			} // else
//		} // while
//		
//		if(user==com) {
//			System.out.println("Congratuation! Try Number: " + count);
//		}

//		while(user>100 || user<1) {
//			System.out.println("You entered: " + user + "\nPlease Enter the number between 1 and 100");
//			user = scan.nextInt();				
//		}		
//		
//		do {
//			++count;
//			if(user==com) {System.out.println("Correct");}
//			else if(user>com) {
//				System.out.println("Under " + user);
//				user = scan.nextInt();
//				}
//			else if(user<com) {
//				System.out.println("Upper " + user);
//				user = scan.nextInt();
//				}
//			else {
//				break;
//				}
//		} while(user!=com); // do-while
//		
//		System.out.println("You tried: " + count);
		
		
	}
}

/*
[문제] 숫자 맞추기 게임
- 컴퓨터가 1 ~ 100사이의 난수를 발생하면, 발생한 난수를 맞추는 게임
- 몇 번만에 맟주었는지 출력한다.

[실행결과]
1 ~ 100사이의 숫자를 맞추세요 (70)

숫자 입력 : 50
50보다 큰 숫자입니다.

숫자 입력 : 85
85보다 작은 숫자입니다.

~~~

숫자 입력 : 70
딩동뎅...x번만에 맞추셨습니다.
 */
