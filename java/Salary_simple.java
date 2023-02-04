import java.util.*;
// 2. java.lang package에 속한 class가 아닐 경우, package import 필요
// Scanner의 경우, jave.util에 속함

public class Salary_simple {

	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
        // 1. Scanner class의 method를 사용하기 위해 Scanner class type variable scan 생성

        int a; // int type variable a 선언
        double b; // double type variable b 선언

        double bonus; // double type variable bonus 선언

        System.out.println("기본급을 (정수로) 입력하세요 : ");

        a = scan.nextInt();
        // int로 선언한 a 변수에 Scanner class를 사용하기 위해 선언한 scan변수를 nextInt() 함수 사용
        // nextInt(): Int type의 입력값을 입력받음
        // 결론: a에 입력된 정수(int)값 저장

		System.out.println("기본급은 " + a + "원 입니다.");

        // 기본급이 100만원 이상일 경우, 기본급의 50%를 보너스로 합산
        // 기본급이 100만원 미만일 경우, 기본급의 60%를 보너스로 합산

        double rate;
        rate = a>=1000000 ? 0.5 : 0.6;
        // 만일 a(입력된 기본급)가 100만원보다 크거나 같으면 double type rate = 0.5이고
        // 만일 a(입력된 기본급)가 100만원보다 크거나 같지 않다면(100만원보다 작다면) double type rate=0.6
        // 하기 if문과 동일
        // if(a>=1000000) rate=0.5;
        // else rate=0.6;

        bonus = a*rate;
        // 보너스 = 변수a(입력된 기본급) * 변수rate(조건식에 따라 결정된 보너스 지급률)
        b = a + bonus;
        // 총계 = 기본급 + 보너스

        System.out.println("총 지급액 : " + (int) b + "원");
        // double type인 변수 b를 (int)를 통해 int type으로 형변환
        System.out.println("추가 보너스 : " + (int) bonus + "원");
        // double type인 변수 bonus를 (int)를 통해 int type으로 형변환

	}

}


/*
Result
기본급을 (정수로) 입력하세요 : 
1000000
기본급은 1000000원 입니다.
총 지급액 : 1500000원
추가 보너스 : 500000원
 */